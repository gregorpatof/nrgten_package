import numpy as np


def get_dv1d(co):
    dv1d = dict()

    dv1d["Xi"] = np.array([0, co["Zk"] - co["Zj"], co["Yj"] - co["Yk"]])
    dv1d["Yi"] = np.array([co["Zj"] - co["Zk"], 0, co["Xk"] - co["Xj"]])
    dv1d["Zi"] = np.array([co["Yk"] - co["Yj"], co["Xj"] - co["Xk"], 0])

    dv1d["Xj"] = np.array([0, co["Zi"] - co["Zk"], co["Yk"] - co["Yi"]])
    dv1d["Yj"] = np.array([co["Zk"] - co["Zi"], 0, co["Xi"] - co["Xk"]])
    dv1d["Zj"] = np.array([co["Yi"] - co["Yk"], co["Xk"] - co["Xi"], 0])

    dv1d["Xk"] = np.array([0, co["Zj"] - co["Zi"], co["Yi"] - co["Yj"]])
    dv1d["Yk"] = np.array([co["Zi"] - co["Zj"], 0, co["Xj"] - co["Xi"]])
    dv1d["Zk"] = np.array([co["Yj"] - co["Yi"], co["Xi"] - co["Xj"], 0])

    dv1d["Xl"] = np.array([0, 0, 0])
    dv1d["Yl"] = np.array([0, 0, 0])
    dv1d["Zl"] = np.array([0, 0, 0])

    return dv1d


def get_dv2d(co):
    dv2d = dict()

    dv2d["Xi"] = np.array([0, 0, 0])
    dv2d["Yi"] = np.array([0, 0, 0])
    dv2d["Zi"] = np.array([0, 0, 0])

    dv2d["Xj"] = np.array([0, co["Zl"] - co["Zk"], co["Yk"] - co["Yl"]])
    dv2d["Yj"] = np.array([co["Zk"] - co["Zl"], 0, co["Xl"] - co["Xk"]])
    dv2d["Zj"] = np.array([co["Yl"] - co["Yk"], co["Xk"] - co["Xl"], 0])

    dv2d["Xk"] = np.array([0, co["Zj"] - co["Zl"], co["Yl"] - co["Yj"]])
    dv2d["Yk"] = np.array([co["Zl"] - co["Zj"], 0, co["Xj"] - co["Xl"]])
    dv2d["Zk"] = np.array([co["Yj"] - co["Yl"], co["Xl"] - co["Xj"], 0])

    dv2d["Xl"] = np.array([0, co["Zk"] - co["Zj"], co["Yj"] - co["Yk"]])
    dv2d["Yl"] = np.array([co["Zj"] - co["Zk"], 0, co["Xk"] - co["Xj"]])
    dv2d["Zl"] = np.array([co["Yk"] - co["Yj"], co["Xj"] - co["Xk"], 0])

    return dv2d


def get_dnorm_v1d(co):
    dnorm_v1d = dict()

    K1 = (co["Yj"] - co["Yi"]) * (co["Zk"] - co["Zj"]) - (co["Yk"] - co["Yj"]) * (co["Zj"] - co["Zi"])
    K2 = (co["Xk"] - co["Xj"]) * (co["Zj"] - co["Zi"]) - (co["Xj"] - co["Xi"]) * (co["Zk"] - co["Zj"])
    K3 = (co["Xj"] - co["Xi"]) * (co["Yk"] - co["Yj"]) - (co["Xk"] - co["Xj"]) * (co["Yj"] - co["Yi"])

    const = 2 * (K1 ** 2 + K2 ** 2 + K3 ** 2) ** 0.5

    dnorm_v1d["Xi"] = (2 * K2 * (co["Zk"] - co["Zj"]) + 2 * K3 * (co["Yj"] - co["Yk"])) / const
    dnorm_v1d["Yi"] = (2 * K1 * (co["Zj"] - co["Zk"]) + 2 * K3 * (co["Xk"] - co["Xj"])) / const
    dnorm_v1d["Zi"] = (2 * K1 * (co["Yk"] - co["Yj"]) + 2 * K2 * (co["Xj"] - co["Xk"])) / const

    dnorm_v1d["Xj"] = (2 * K2 * (co["Zi"] - co["Zk"]) + 2 * K3 * (co["Yk"] - co["Yi"])) / const
    dnorm_v1d["Yj"] = (2 * K1 * (co["Zk"] - co["Zi"]) + 2 * K3 * (co["Xi"] - co["Xk"])) / const
    dnorm_v1d["Zj"] = (2 * K1 * (co["Yi"] - co["Yk"]) + 2 * K2 * (co["Xk"] - co["Xi"])) / const

    dnorm_v1d["Xk"] = (2 * K2 * (co["Zj"] - co["Zi"]) + 2 * K3 * (co["Yi"] - co["Yj"])) / const
    dnorm_v1d["Yk"] = (2 * K1 * (co["Zi"] - co["Zj"]) + 2 * K3 * (co["Xj"] - co["Xi"])) / const
    dnorm_v1d["Zk"] = (2 * K1 * (co["Yj"] - co["Yi"]) + 2 * K2 * (co["Xi"] - co["Xj"])) / const

    dnorm_v1d["Xl"] = 0
    dnorm_v1d["Yl"] = 0
    dnorm_v1d["Zl"] = 0

    return dnorm_v1d


def get_dnorm_v2d(co):
    dnorm_v2d = dict()

    L1 = (co["Yk"] - co["Yj"]) * (co["Zl"] - co["Zk"]) - (co["Yl"] - co["Yk"]) * (co["Zk"] - co["Zj"])
    L2 = (co["Xl"] - co["Xk"]) * (co["Zk"] - co["Zj"]) - (co["Xk"] - co["Xj"]) * (co["Zl"] - co["Zk"])
    L3 = (co["Xk"] - co["Xj"]) * (co["Yl"] - co["Yk"]) - (co["Xl"] - co["Xk"]) * (co["Yk"] - co["Yj"])

    const = 2 * (L1 ** 2 + L2 ** 2 + L3 ** 2) ** 0.5

    dnorm_v2d["Xi"] = 0
    dnorm_v2d["Yi"] = 0
    dnorm_v2d["Zi"] = 0

    dnorm_v2d["Xj"] = (2 * L2 * (co["Zl"] - co["Zk"]) + 2 * L3 * (co["Yk"] - co["Yl"])) / const
    dnorm_v2d["Yj"] = (2 * L1 * (co["Zk"] - co["Zl"]) + 2 * L3 * (co["Xl"] - co["Xk"])) / const
    dnorm_v2d["Zj"] = (2 * L1 * (co["Yl"] - co["Yk"]) + 2 * L2 * (co["Xk"] - co["Xl"])) / const

    dnorm_v2d["Xk"] = (2 * L2 * (co["Zj"] - co["Zl"]) + 2 * L3 * (co["Yl"] - co["Yj"])) / const
    dnorm_v2d["Yk"] = (2 * L1 * (co["Zl"] - co["Zj"]) + 2 * L3 * (co["Xj"] - co["Xl"])) / const
    dnorm_v2d["Zk"] = (2 * L1 * (co["Yj"] - co["Yl"]) + 2 * L2 * (co["Xl"] - co["Xj"])) / const

    dnorm_v2d["Xl"] = (2 * L2 * (co["Zk"] - co["Zj"]) + 2 * L3 * (co["Yj"] - co["Yk"])) / const
    dnorm_v2d["Yl"] = (2 * L1 * (co["Zj"] - co["Zk"]) + 2 * L3 * (co["Xk"] - co["Xj"])) / const
    dnorm_v2d["Zl"] = (2 * L1 * (co["Yk"] - co["Yj"]) + 2 * L2 * (co["Xj"] - co["Xk"])) / const

    return dnorm_v2d
