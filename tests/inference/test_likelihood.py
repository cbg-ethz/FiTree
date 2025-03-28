import jax
import numpy as np

from fitree._trees._wrapper import VectorizedTrees
from fitree._inference._likelihood import _mlogp

tol = 1e-3
C_0 = 1e5
beta = 8.8
t = 60

def test_mlogp_1():
    
    # Test case 1: lambda_1 = 0, lambda_2 > 0
    expected_output = np.array([
        0.00296105481930962573679264846772817142200468794895707272472828500876831104041511578068343321459626535,
        0.002711784462836684232322813249328352780312706814478932160655950884287557685966659465298998096395924892,
        0.00247333554425904403688922995001807192554192076718485239695277749656116458776251761091313801867771381,
        0.002245738153051455529192524419274661796295427514417637428381628030942727128038696986125711034338324306,
        0.002029021058437020390179188908386982525427099981665687150081521096828870405380177539699957493240717131,
        0.001823211698938432271265432546906182321111344639670767946674748299909969110159750768060498606664339627,
        0.001628336172386190135422684788135976661163271176656990067806560881015219374653057844183913702366410692,
        0.001444419226387936699336604241680821534413108766735586401257059673381761031964851493378053146738871722,
        0.001271484249256174785096425737075375225657779251692207397476589296434784068942658254064028647777949521,
        0.001109553261370454410688038383269639164584475987319134499359364729306488118820845885754926262303878567,
        0.0009586469068838665980618553662676085444699042802585369741382430219081650317117508803269073233673074933,
        0.000818784445474775041446574239790825466289835394707530584289378591388402535445520416322547156964034739,
        0.0006899837431845768595454845869674748595423964412436056756555508989188603311523539524878521296558226418,
        0.0005722612592947915549967446930027041289195588804541777134638979199540427102300671358290884166653833943,
        0.0004656320195944772196984034555482903695783140676151826557772629509494337281942414520762077617283745883,
    ])

    N_trees = expected_output.shape[0]
    vec_trees = VectorizedTrees(
        cell_number=np.array(
            [(0.0, np.power(10, i / 2)) for i in range(N_trees)]
        ),
        seq_cell_number=np.array(
            [(0.0, np.power(10, i / 2)) for i in range(N_trees)]
        ),
        observed=np.array([(1, 1)] * N_trees),
        sampling_time=np.array([t] * N_trees),
        weight=np.array([1.0] * N_trees),
        tumor_size=np.array([1.0] * N_trees),
        node_id=np.array([0, 1], dtype=np.int32),
        parent_id=np.array([-1, 0], dtype=np.int32),
        alpha=np.array([beta, 9.2]),
        nu=np.array([3.1 / C_0, 1e-5]),
        lam=np.array([0.0, 9.2 - beta]),
        rho=np.array([3.1 / C_0 / beta, 1e-5 / 9.2]),
        phi=np.array([beta, 9.2 / (9.2 - beta)]),
        delta=np.array([0.0, 9.2 - beta]),
        r=np.array([2.0, 1.0]),
        gamma=np.array([0.0, 0.0]),
        N_trees=N_trees,
        N_patients=1,
        n_nodes=2,
        beta=beta,
        C_s=1e9,
        C_0=C_0,
        genotypes=np.array([[True, False], [True, True]]),
        t_max=100,
    )

    actual_output = jax.vmap(
        _mlogp,
        in_axes=(
            VectorizedTrees(
                0,  # cell_number
                0,  # seq_cell_number
                0,  # observed
                0,  # sampling_time
                0,  # weight
                0,  # tumor_size
                None,  # node_id
                None,  # parent_id
                None,  # alpha
                None,  # nu
                None,  # lam
                None,  # rho
                None,  # phi
                None,  # delta
                None,  # r
                None,  # gamma
                None,  # N_trees
                None,  # N_patients
                None,  # n_nodes
                None,  # beta
                None,  # C_s
                None,  # C_0
                None,  # genotypes
                None,  # t_max
            ),
            None,
            None,
            None,
            None,
            None,
        ),
    )(vec_trees, 1, np.inf, 1e-16, False, 0.01)

    assert np.allclose(actual_output, np.log(1.0 - expected_output), atol=tol)
    

def test_mlogp_2():
    
    # Test case 1: lambda_1 < 0, lambda_2 > 0
    expected_output = np.array([
        0.0002982551013232323852061365494142036114232162147212408127489585912618204300392823946428234548214619394,
        0.0002853276919176614151462789479319843854740069997938411517163701963023890096675134886769254625984735146,
        0.0002724001152653128825036550864714206146005459712548720638312815289791863429730687800133309208304602532,
        0.0002594723713642083566678484602564969779332351305542696668495203913684574929929973978110364609743937697,
        0.0002465444602127722828594541557902147604072579075756563582572600303216572509469983141568934674894406817,
        0.0002336163818107031886367821026227802619901976198607266717352433874729731820856790046699172388777804199,
        0.0002206881361617287164095843365905256397587091554498708740699509359636588984359980188671829039788146421,
        0.0002077597232823179138932967668237299208805693502642686375989786640078848563877400776201657198099313321,
        0.0001948311432292322717105623869382777112256531873502132420825471878232895411002112441592584465861115601,
        0.0001819023961866508782170547965937809147567217254881971088963227926981158469903661276118599733689300505,
        0.0001689734827416878841376245445352169957782683058233962036772918370798441967733096820175722470835181169,
        0.0001560444047556661924409846419012006560432874445641943517601190780765257642479829968422237470492410454,
        0.0001431151681193593749634759257024589358435210840087523810167364171970402529102701044896686431212834154,
        0.0001301857914659138150799498174231746989769455871992587457940383580838526416587145419236516141258628272,
        0.0001172563337235788114144807807466922298351307809174346286304877403597956900497053699411095716192211117,
    ])

    N_trees = expected_output.shape[0]
    vec_trees = VectorizedTrees(
        cell_number=np.array(
            [(0.0, np.power(10, i / 2)) for i in range(N_trees)]
        ),
        seq_cell_number=np.array(
            [(0.0, np.power(10, i / 2)) for i in range(N_trees)]
        ),
        observed=np.array([(1, 1)] * N_trees),
        sampling_time=np.array([t] * N_trees),
        weight=np.array([1.0] * N_trees),
        tumor_size=np.array([1.0] * N_trees),
        node_id=np.array([0, 1], dtype=np.int32),
        parent_id=np.array([-1, 0], dtype=np.int32),
        alpha=np.array([8.5, 9.2]),
        nu=np.array([3.1 / C_0, 1e-5]),
        lam=np.array([8.5 - beta, 9.2 - beta]),
        rho=np.array([3.1 / C_0 / 8.5, 1e-5 / 9.2]),
        phi=np.array([-beta / (8.5 - beta), 9.2 / (9.2 - beta)]),
        delta=np.array([0.0, 9.2 - beta]),
        r=np.array([1.0, 1.0]),
        gamma=np.array([0.0, 0.0]),
        N_trees=N_trees,
        N_patients=1,
        n_nodes=2,
        beta=beta,
        C_s=1e9,
        C_0=C_0,
        genotypes=np.array([[True, False], [True, True]]),
        t_max=100,
    )

    actual_output = jax.vmap(
        _mlogp,
        in_axes=(
            VectorizedTrees(
                0,  # cell_number
                0,  # seq_cell_number
                0,  # observed
                0,  # sampling_time
                0,  # weight
                0,  # tumor_size
                None,  # node_id
                None,  # parent_id
                None,  # alpha
                None,  # nu
                None,  # lam
                None,  # rho
                None,  # phi
                None,  # delta
                None,  # r
                None,  # gamma
                None,  # N_trees
                None,  # N_patients
                None,  # n_nodes
                None,  # beta
                None,  # C_s
                None,  # C_0
                None,  # genotypes
                None,  # t_max
            ),
            None,
            None,
            None,
            None,
            None,
        ),
    )(vec_trees, 1, np.inf, 1e-16, False, 0.01)

    assert np.allclose(actual_output, np.log(1.0 - expected_output), atol=tol)