# Third Party Notices

The DreamerV3 source retains its original [MIT license](LICENSE) and attribution
to Danijar Hafner and contributors. Craftax is an external dependency; its
license and asset notices remain with the upstream package.

The optional WUJI Craftax ports import NSGA-II selection and crossover from the
upstream WUJI checkout at `.deps/wuji_upstream`. The pinned upstream commit in
`dreamerv3/wuji_craftax.py` is `de7161c29c59c4cc9ffb729d8c19dfb52e37a2c6`.
The upstream WUJI code is GPL-3.0-or-later; its license must be preserved when
that code is obtained or redistributed. The upstream checkout is not vendored
in this release. The repository's Dreamer MIT notice does not replace WUJI's
license. The A2C port additionally uses Stable-Baselines3 and PyTorch under
their respective upstream licenses.

Neither Craftax nor DreamerV3 is claimed as an original contribution of TEA.
