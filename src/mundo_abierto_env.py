"""Mundo abierto: un solo cuerpo de mosca con patas + alas + probóscide activas.

Une lo que flybody trae separado (vuelo sin patas, caminata sin alas) en una física
única, y expone un controlador de bajo nivel que:
  * construye las observaciones que esperan las políticas de Janelia (caminar / volar)
    a partir de la física, mapeando articulaciones y actuadores POR NOMBRE;
  * genera un "fantasma" de referencia en tiempo real (rumbo, velocidad, altura) para
    que las políticas lo sigan — así el cerebro puede decidir a dónde ir;
  * maneja alas (generador de aleteo WPG), patas retraídas en vuelo, alas plegadas en
    tierra, y la probóscide por posición.
"""
import numpy as np
from dm_control import composer
from dm_control.locomotion.arenas import floors

from flybody.fruitfly import fruitfly
from flybody.quaternions import get_dquat_local
from flybody.tasks.base import FruitFlyTask
from flybody.tasks.constants import _WING_PARAMS
from flybody.tasks.pattern_generators import WingBeatPatternGenerator
from flybody.utils import any_substr_in_str

PHYS_DT = 5e-5
CTRL_DT = 2e-4           # paso de control (vuelo); la caminata decide cada 10 pasos
WALK_EVERY = 10          # -> 2 ms
Z_WALK = 0.1278          # altura del tórax al caminar (cm)
BODY_PITCH = 47.5        # grados, postura de vuelo

# Listas de nombres que esperan las políticas (orden = orden de sus entornos originales)
WALK_JOINTS = ['head_abduct', 'head_twist', 'head', 'abdomen_abduct', 'abdomen', 'abdomen_abduct_2', 'abdomen_2', 'abdomen_abduct_3', 'abdomen_3', 'abdomen_abduct_4', 'abdomen_4', 'abdomen_abduct_5', 'abdomen_5', 'abdomen_abduct_6', 'abdomen_6', 'abdomen_abduct_7', 'abdomen_7', 'haltere_left', 'haltere_right'] + [
    f'{j}_{s}_{side}' for s in ['T1', 'T2', 'T3'] for side in ['left', 'right']
    for j in ['coxa_abduct', 'coxa_twist', 'coxa', 'femur_twist', 'femur', 'tibia', 'tarsus', 'tarsus2', 'tarsus3', 'tarsus4', 'tarsus5']]
FLY_JOINTS = ['head_abduct', 'head_twist', 'head', 'wing_yaw_left', 'wing_roll_left', 'wing_pitch_left', 'wing_yaw_right', 'wing_roll_right', 'wing_pitch_right', 'abdomen_abduct', 'abdomen', 'abdomen_abduct_2', 'abdomen_2', 'abdomen_abduct_3', 'abdomen_3', 'abdomen_abduct_4', 'abdomen_4', 'abdomen_abduct_5', 'abdomen_5', 'abdomen_abduct_6', 'abdomen_6', 'abdomen_abduct_7', 'abdomen_7', 'haltere_left', 'haltere_right']
# acción de la política de caminata: clases en orden adhesion, head, abdomen, legs
WALK_ACTIONS = [f'adhere_claw_{s}_{side}' for s in ['T1', 'T2', 'T3'] for side in ['left', 'right']] + \
    ['head_abduct', 'head_twist', 'head', 'abdomen_abduct', 'abdomen'] + [
    f'{j}_{s}_{side}' for s in ['T1', 'T2', 'T3'] for side in ['left', 'right']
    for j in ['coxa_abduct', 'coxa_twist', 'coxa', 'femur_twist', 'femur', 'tibia', 'tarsus', 'tarsus2']]
# orden de actuator_activation observado por la política de caminata = orden MJCF:
WALK_ACT_OBS = ['head_abduct', 'head_twist', 'head', 'abdomen_abduct', 'abdomen'] + [
    f'{j}_{s}_{side}' for s in ['T1', 'T2', 'T3'] for side in ['left', 'right']
    for j in ['coxa_abduct', 'coxa_twist', 'coxa', 'femur_twist', 'femur', 'tibia', 'tarsus', 'tarsus2']] + \
    [f'adhere_claw_{s}_{side}' for s in ['T1', 'T2', 'T3'] for side in ['left', 'right']]
# acción de la política de vuelo: head, wings, abdomen, user(frecuencia)
FLY_ACTIONS = ['head_abduct', 'head_twist', 'head', 'wing_yaw_left', 'wing_roll_left', 'wing_pitch_left',
               'wing_yaw_right', 'wing_roll_right', 'wing_pitch_right', 'abdomen_abduct', 'abdomen']
WING_JOINTS = ['wing_yaw_left', 'wing_roll_left', 'wing_pitch_left', 'wing_yaw_right', 'wing_roll_right', 'wing_pitch_right']
CLAWS = ['claw_T1_left', 'claw_T1_right', 'claw_T2_left', 'claw_T2_right', 'claw_T3_left', 'claw_T3_right']
MOUTH = ['rostrum', 'haustellum_abduct', 'haustellum', 'labrum_left', 'labrum_right']


class MundoAbierto(FruitFlyTask):
    def __init__(self, food_positions=(), time_limit=60.0):
        arena = floors.Floor(size=(60, 60), reflectance=0.1)
        super().__init__(walker=fruitfly.FruitFly, arena=arena, time_limit=time_limit,
                         use_legs=True, use_wings=True, use_mouth=True, use_antennae=False,
                         physics_timestep=PHYS_DT, control_timestep=CTRL_DT,
                         joint_filter=0.01, adhesion_filter=0.007, num_user_actions=0,
                         body_pitch_angle=BODY_PITCH)
        model = self._walker.mjcf_model
        # --- configuración de vuelo (copiada de flybody.tasks.base.Flying) ---
        for i, dclass in enumerate(['yaw', 'roll', 'pitch']):
            model.find('default', dclass).general.gainprm[0] = _WING_PARAMS['gainprm'][i]
        for geom in model.find_all('geom'):
            if 'fluid' in geom.name:
                geom.fluidshape = 'ellipsoid'
                geom.fluidcoef = _WING_PARAMS['fluidcoef']
        wing_default_joint = model.find('default', 'wing').joint
        wing_default_joint.stiffness = _WING_PARAMS['stiffness']
        wing_default_joint.damping = _WING_PARAMS['damping']
        # las alas no colisionan (plegadas tocarían el piso y voltearían a la mosca)
        for geom in model.find_all('geom'):
            if 'wing' in geom.name:
                geom.contype = 0
                geom.conaffinity = 0
        # las alas NO deben llevar filtro de 10 ms (aletean a 218 Hz)
        for act in model.find_all('actuator'):
            if 'wing' in act.name:
                act.dyntype = None
                act.dynprm = None
        contact = model.contact
        for body in model.find_all('body'):
            if any_substr_in_str(['coxa', 'femur', 'tibia', 'tarsus', 'claw'], body.name):
                for wing in ['wing_left', 'wing_right']:
                    contact.add('exclude', name=f'{body.name}_{wing}', body1=body.name, body2=wing)
        # --- configuración de caminata (copiada de Walking) ---
        for geom in arena.ground_geoms:
            geom.friction = (0.5,)
            geom.solref = (0.001, 1)
            geom.solimp = (0.95, 0.99, 0.01)
        model.find('default', 'adhesion-collision').geom.friction = (1.0,)
        # springrefs (posiciones de reposo) de patas y alas
        self.leg_joints, self.leg_springrefs, self.wing_joints, self.wing_springrefs = [], [], [], []
        for joint in model.find_all('joint'):
            sr = joint.springref
            if sr is None and joint.dclass is not None:
                sr = joint.dclass.joint.springref
            sr = sr or 0.
            if any_substr_in_str(['coxa', 'femur', 'tibia', 'tarsus'], joint.name):
                self.leg_joints.append(joint); self.leg_springrefs.append(sr)
            if any_substr_in_str(['yaw', 'roll', 'pitch'], joint.name):
                self.wing_joints.append(joint); self.wing_springrefs.append(sr)
        self.wing_springrefs = np.array(self.wing_springrefs)
        # --- objetos del mundo: comida (visual) y depredador (mocap) ---
        world = arena.mjcf_model.worldbody
        for i, p in enumerate(food_positions):
            world.add('geom', name=f'food_{i}', type='sphere', size=(0.12,), pos=(p[0], p[1], 0.06),
                      rgba=(0.95, 0.75, 0.1, 1), contype=0, conaffinity=0)
            world.add('geom', name=f'food_stain_{i}', type='cylinder', size=(0.45, 0.005),
                      pos=(p[0], p[1], 0.003), rgba=(0.9, 0.6, 0.1, 0.35), contype=0, conaffinity=0)
        pred = world.add('body', name='predator', mocap=True, pos=(0, 0, -5))
        pred.add('geom', name='predator_geom', type='sphere', size=(1.0,), rgba=(0.15, 0.1, 0.1, 1),
                 contype=0, conaffinity=0)
        world.add('light', pos=(0, 0, 30), dir=(0, 0, -1), diffuse=(0.8, 0.8, 0.8), castshadow=False)

    def get_reward_factors(self, physics):
        return (1.,)

    def check_termination(self, physics):
        return False


class Cuerpo:
    """Controlador de bajo nivel: observaciones para las políticas y composición de acciones."""

    def __init__(self, food_positions=()):
        self.task = MundoAbierto(food_positions=food_positions)
        self.env = composer.Environment(task=self.task, time_limit=1e9, strip_singleton_obs_buffer_dim=True)
        self.env.reset()
        self.physics = self.env.physics
        self.walker = self.task.walker
        m = self.physics.model
        self.wpg = WingBeatPatternGenerator()  # patrón aproximado de aleteo (sin dataset)
        self.wpg.reset()

        # --- índices por nombre ---
        joint_names = [j.name for j in self.walker.observable_joints]
        self.all_joints = self.walker.observable_joints
        self.walk_j = np.array([joint_names.index(n) for n in WALK_JOINTS])
        self.fly_j = np.array([joint_names.index(n) for n in FLY_JOINTS])
        act_names = [a.name for a in self.walker.actuators]
        self.act_id = {n: i for i, n in enumerate(act_names)}
        self.walk_a = np.array([self.act_id[n] for n in WALK_ACTIONS])
        self.fly_a = np.array([self.act_id[n] for n in FLY_ACTIONS])
        self.wing_a = np.array([self.act_id[n] for n in WING_JOINTS])
        self.mouth_a = np.array([self.act_id[n] for n in MOUTH])
        self.leg_a = np.array([self.act_id[j.name] for j in self.task.leg_joints if j.name in self.act_id])
        self.leg_spring_ctrl = np.array([sr for j, sr in zip(self.task.leg_joints, self.task.leg_springrefs)
                                         if j.name in self.act_id])
        self.walk_actadr = m.actuator_actadr[[self.act_id[n] for n in WALK_ACT_OBS]]
        self.sens_hist = []   # sensordata de los últimos pasos de control (promedio de 2 ms)
        self.ctrlrange = m.actuator_ctrlrange.copy()
        self.sensor = {m.sensor(i).name.split('/')[-1]: i for i in range(m.nsensor)}
        self.claw_sites = [self.walker.mjcf_model.find('site', c) for c in CLAWS] + [self.walker.head_site]
        self.root = self.task._root_joint
        self.nu = m.nu

    # --- estado ---
    def pose(self):
        p, q = self.walker.get_pose(self.physics)
        return np.array(p, float), np.array(q, float)

    def heading(self):
        _, q = self.pose()
        w, x, y, z = q
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _sens(self, name, mean=False):
        i = self.sensor[name]
        adr, dim = self.physics.model.sensor_adr[i], self.physics.model.sensor_dim[i]
        if mean and self.sens_hist:
            return np.mean([h[adr:adr + dim] for h in self.sens_hist], axis=0)
        return self.physics.data.sensordata[adr:adr + dim].copy()

    def touch(self):
        return np.array([self._sens(f'touch_claw_{s}_{side}')[0] for s in ['T1', 'T2', 'T3'] for side in ['left', 'right']])

    def thorax_height(self):
        return float(self.physics.bind(self.walker.thorax).xpos[2])

    def upright(self):
        return float(np.reshape(self.physics.bind(self.walker.root_body).xmat, (3, 3))[2, 2])

    # --- observaciones ---
    def _obs_comun(self, jidx, mean=False):
        ph = self.physics
        d = {
            'walker/accelerometer': self._sens('accelerometer', mean),
            'walker/gyro': self._sens('gyro', mean),
            'walker/velocimeter': self._sens('velocimeter', mean),
            'walker/world_zaxis': np.reshape(ph.bind(self.walker.root_body).xmat, (3, 3))[2].copy(),
            'walker/joints_pos': ph.bind(self.all_joints).qpos[jidx].copy(),
            'walker/joints_vel': ph.bind(self.all_joints).qvel[jidx].copy(),
        }
        return d

    def _ref(self, ref_qpos):
        fly_pos, fly_quat = self.pose()
        disp = self.walker.transform_vec_to_egocentric_frame(self.physics, ref_qpos[:, :3] - fly_pos)
        return np.asarray(disp), np.asarray(get_dquat_local(fly_quat, ref_qpos[:, 3:7]))

    def obs_caminar(self, ref_qpos):
        d = self._obs_comun(self.walk_j, mean=True)
        d['walker/actuator_activation'] = self.physics.data.act[self.walk_actadr].copy()
        d['walker/force'] = np.concatenate([self._sens(f'force_tarsus_{s}_{side}', True) for s in ['T1', 'T2', 'T3'] for side in ['left', 'right']])
        d['walker/touch'] = np.array([self._sens(f'touch_claw_{s}_{side}', True)[0] for s in ['T1', 'T2', 'T3'] for side in ['left', 'right']])
        app = self.physics.bind(self.claw_sites).xpos
        torso_pos = self.physics.bind(self.walker.root_body).xpos
        torso_mat = np.reshape(self.physics.bind(self.walker.root_body).xmat, (3, 3))
        d['walker/appendages_pos'] = np.reshape(np.dot(app - torso_pos, torso_mat), -1)
        d['walker/ref_displacement'], d['walker/ref_root_quat'] = self._ref(ref_qpos)
        return d

    def obs_volar(self, ref_qpos):
        d = self._obs_comun(self.fly_j)
        d['walker/actuator_activation'] = np.zeros(0, np.float32)
        d['walker/ref_displacement'], d['walker/ref_root_quat'] = self._ref(ref_qpos)
        return d

    # --- acciones ---
    def ctrl_base(self):
        return np.zeros(self.nu)

    def poner_caminata(self, ctrl, walk_action):
        ctrl[self.walk_a] = walk_action

    def patas_retraidas(self, ctrl):
        ctrl[self.leg_a] = self.leg_spring_ctrl

    def alas_plegadas(self, ctrl):
        qpos = self.physics.bind(self.task.wing_joints).qpos
        ctrl[self.wing_a] = self.task.wing_springrefs - qpos

    def alas_volando(self, ctrl, fly_action, freq_action=0.0):
        ctrl[self.fly_a] = fly_action
        f = self.wpg.base_beat_freq * (1 + self.wpg.rel_freq_range * freq_action)
        target = self.wpg.step(ctrl_freq=f)
        qpos = self.physics.bind(self.task.wing_joints).qpos
        ctrl[self.wing_a] += target - qpos

    def probóscide(self, ctrl, ext):
        ctrl[self.mouth_a] = [-1.2 * ext, 0.0, -1.5 * ext, 1.0 * ext, 1.0 * ext]

    def paso(self, ctrl):
        ctrl = np.clip(ctrl, self.ctrlrange[:, 0], self.ctrlrange[:, 1])
        self.physics.set_control(ctrl)
        for _ in range(int(round(CTRL_DT / PHYS_DT))):
            self.physics.step()
        self.sens_hist.append(self.physics.data.sensordata.copy())
        if len(self.sens_hist) > WALK_EVERY:
            self.sens_hist.pop(0)

    # --- transiciones (simplificaciones documentadas) ---
    def despegar(self, v_fwd=15.0, z_salto=0.5, vz=4.0):
        """Salto de despegue: la mosca queda a z_salto con cabeceo de vuelo y velocidad
        (v_fwd, vz); patas retraídas y alas en fase inicial. Devuelve el fantasma de vuelo."""
        ph = self.physics
        p, _ = self.pose(); yaw = self.heading()
        self.walker.set_pose(ph, np.array([p[0], p[1], z_salto]), quat_heading(yaw, -BODY_PITCH))
        self.walker.set_velocity(ph, np.array([v_fwd * np.cos(yaw), v_fwd * np.sin(yaw), vz]))
        ph.bind(self.task.leg_joints).qpos = self.task.leg_springrefs
        ph.bind(self.task.leg_joints).qvel = 0.0
        qw, vw = self.wpg.reset(initial_phase=0.0, return_qvel=True)
        ph.bind(self.task.wing_joints).qpos = qw
        ph.bind(self.task.wing_joints).qvel = vw
        ph.forward()
        return Fantasma([p[0], p[1]], yaw, z_salto, pitch_deg=-BODY_PITCH)

    def nivelar(self, frenado=0.0):
        """Aterrizaje simplificado: nivela el cuerpo, extiende las patas a la postura de
        apoyo (la misma con la que la mosca aparece al inicio) y frena, conservando
        posición horizontal y rumbo. La política de caminata se encarga del resto."""
        ph = self.physics
        p, _ = self.pose(); yaw = self.heading()
        v = ph.bind(self.root).qvel[:3].copy()
        self.walker.set_pose(ph, np.array([p[0], p[1], Z_WALK + 0.03]), quat_heading(yaw, 0.0))
        self.walker.set_velocity(ph, np.array([v[0] * frenado, v[1] * frenado, 0.0]), np.zeros(3))
        ph.bind(self.task.wing_joints).qpos = self.task.wing_springrefs   # alas plegadas al instante
        ph.bind(self.task.wing_joints).qvel = 0.0
        ph.bind(self.task.leg_joints).qpos = 0.0
        ph.bind(self.task.leg_joints).qvel = 0.0
        ph.data.act[:] = 0.0
        ph.forward()

    # --- mundo ---
    def mover_depredador(self, pos):
        self.physics.named.data.mocap_pos['predator'] = pos

    def render(self, camera, w=640, h=480):
        return self.physics.render(camera_id=camera, width=w, height=h)


# ---------------------------------------------------------------------------
def quat_heading(yaw, pitch_deg=0.0):
    """Cuaternión con cabeceo (sobre y) y luego rumbo (sobre z)."""
    p = np.deg2rad(pitch_deg)
    qp = np.array([np.cos(p / 2), 0, np.sin(p / 2), 0])
    qy = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    w1, x1, y1, z1 = qy; w2, x2, y2, z2 = qp
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


class Fantasma:
    """Referencia que las políticas siguen: un punto con rumbo, velocidad y altura."""

    def __init__(self, pos, yaw, z, pitch_deg=0.0):
        self.pos = np.array(pos, float); self.yaw = float(yaw); self.z = float(z)
        self.pitch = pitch_deg

    def futuro(self, n, dt, speed, yaw_rate, z_rate=0.0):
        """Avanza el fantasma un paso y devuelve n poses futuras (n,7)."""
        out = np.zeros((n, 7))
        pos, yaw, z = self.pos.copy(), self.yaw, self.z
        for k in range(n):
            if k > 0:
                yaw += yaw_rate * dt
                pos = pos + speed * dt * np.array([np.cos(yaw), np.sin(yaw)])
                z += z_rate * dt
            out[k, :2] = pos; out[k, 2] = z; out[k, 3:] = quat_heading(yaw, self.pitch)
        # el fantasma real avanza un paso
        self.yaw += yaw_rate * dt
        self.pos += speed * dt * np.array([np.cos(self.yaw), np.sin(self.yaw)])
        self.z += z_rate * dt
        return out
