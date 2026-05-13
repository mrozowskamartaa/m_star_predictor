import numpy as np
from typing import Union, Literal


def calculate_f(latitude:  Union[float, np.ndarray]) ->  Union[float, np.ndarray]:
    omega = 2 * np.pi / 24 / 60 / 60
    return 2 * omega * np.sin(latitude * np.pi / 180)


def calculate_T(latitude:  Union[float, np.ndarray]) ->  Union[float, np.ndarray]:
    return 2 * np.pi / calculate_f(latitude)


def compute_u_star(
        tau: Union[float, np.ndarray],
        rho: float = 1027
) ->  Union[float, np.ndarray]:
    return (tau / rho) ** 0.5


def calculate_analytical_m_star_N(
        H: np.ndarray,
        f: Union[float, np.ndarray],
        u_star: Union[float, np.ndarray],
        c_N1: float = 0.275, 
        c_N2: float = 8.0, 
        c_N3: float = 5.0
) -> np.ndarray:
    x = H*f/u_star
    return c_N1 * (1 - (1 + c_N2 * np.exp(-c_N3*x)) ** -1)


def calculate_analytical_m_star_Nb(
        H: np.ndarray,
        f: Union[float, np.ndarray],
        u_star: Union[float, np.ndarray],
        c_Nb1: float = 0.5, 
        c_Nb2: float = 3.0
) -> np.ndarray:
    x = H*f/u_star
    return c_Nb1 * np.exp(-c_Nb2 * x)


def calculate_analytical_m_star_S(
        H: np.ndarray,
        f: Union[float, np.ndarray],
        u_star: Union[float, np.ndarray],
        B: float,
        c_S1: float = 0.2,
        c_S2: float = 0.4
) -> np.ndarray:
    return c_S1 * (B**2 * H / (u_star**5 * f)) ** c_S2


def calculate_Psi(
        m_star: np.ndarray,
        u_star: Union[float, np.ndarray],
        B: Union[float, np.ndarray],
        H: Union[float, np.ndarray],
        c_psi: float = 0.67
) -> np.ndarray:
    return (2 * m_star * u_star ** 3) / (c_psi * B * H + 2 * m_star * u_star**3)


def calculate_M(
        H: np.ndarray,
        f: Union[float, np.ndarray],
        u_star: Union[float, np.ndarray],
        B: Union[float, np.ndarray],
        wb: np.ndarray,
        neutral_mode: Literal["N", "Nb"],
        n_star: float = 0.066
) -> np.ndarray:
    neutral = np.where(np.logical_and(B > -1e-8, B < 1e-8))
    stabilizing = np.where(B > 1e-8)
    destabilizing = np.where(B < -1e-8)

    if neutral_mode == "N":
        m_star_neutral = calculate_analytical_m_star_N(
            H=H[neutral],
            f=f[neutral],
            u_star=u_star[neutral]
        )

        m_star_destabilizing = calculate_analytical_m_star_N(
            H=H[destabilizing],
            f=f[destabilizing],
            u_star=u_star[destabilizing]
        )

    elif neutral_mode == "Nb":
        m_star_neutral = calculate_analytical_m_star_Nb(
            H=H[neutral],
            f=f[neutral],
            u_star=u_star[neutral]
        )

        m_star_destabilizing = calculate_analytical_m_star_Nb(
            H=H[destabilizing],
            f=f[destabilizing],
            u_star=u_star[destabilizing]
        )

    m_star_stabilizing = calculate_analytical_m_star_S(
        H=H[stabilizing],
        f=f[stabilizing],
        u_star=u_star[stabilizing],
        B=B[stabilizing]
    )

    M = np.empty(len(B))
    M[neutral] = m_star_neutral * u_star[neutral]**3
    M[stabilizing] = m_star_stabilizing * u_star[stabilizing]**3
    Psi = calculate_Psi(
        m_star=m_star_destabilizing,
        u_star=u_star[destabilizing],
        B=B[destabilizing],
        H=H[destabilizing]
    )
    M[destabilizing] = m_star_destabilizing * Psi + n_star * wb[destabilizing]

    return M
