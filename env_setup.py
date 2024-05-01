from mimopy import AntennaArray, Network
from mimopy.channels import LoS, Rician, SphericalWave
from maenv import MIMOMAEnv


def FD_network(args):
    tx = AntennaArray.ula(
        args.Nt, name="tx", array_center=args.tx_coord
    )
    rx = AntennaArray.ula(
        args.Nr, name="rx", array_center=args.rx_coord
    )
    dl = AntennaArray.ula(
        args.Ndl, name="dl", array_center=args.dl_coord
    )
    ul = AntennaArray.ula(
        args.Nul, name="ul", array_center=args.ul_coord
    )
    tx.power = args.tx_power
    rx.power = args.rx_power
    dl.power = args.dl_power
    ul.power = args.ul_power

    l_tx_dl = LoS(tx, dl, "tx-dl").realize()
    # l_ul_rx = LoS(ul, rx, "ul-rx").realize()
    l_ul_rx = LoS(rx, ul, "rx-ul").realize()
    l_si = LoS(tx, rx, "si").realize()

    net = Network("FD Network")
    net.add_links([l_tx_dl, l_ul_rx])
    net.add_target_links([l_tx_dl, l_ul_rx])
    net.add_target_nodes([tx, rx])

    env = MIMOMAEnv(net, args.target_meas, args.metrics)
    env.reset(args.seed)
    return env


def DL_network(args):
    tx = AntennaArray.ula(
        args.Nt, name="tx", array_center=args.tx_coord
    )
    dl = AntennaArray.ula(
        args.Ndl, name="dl", array_center=args.dl_coord
    )
    tx.power = args.tx_power
    dl.power = args.dl_power
    l_tx_dl = LoS(tx, dl, "los").realize()
    net = Network("DL Network")
    net.add_target_links([l_tx_dl])
    net.add_target_nodes([tx])

    env = MIMOMAEnv(net, args.target_meas, args.metrics)
    env.reset(args.seed)
    return env


def UL_network(args):
    rx = AntennaArray.ula(
        args.Nr, name="rx", array_center=args.rx_coord
    )
    ul = AntennaArray.ula(
        args.Nul, name="ul", array_center=args.ul_coord
    )
    rx.power = args.rx_power
    ul.power = args.ul_power
    l_ul_rx = LoS(ul, rx, "los").realize()
    net = Network("UL Network")
    net.add_target_links([l_ul_rx])
    net.add_target_nodes([rx])

    env = MIMOMAEnv(net, args.target_meas, args.metrics)
    env.reset(args.seed)
    return env