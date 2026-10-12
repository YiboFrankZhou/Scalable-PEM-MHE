# Third-party code

Third-party implementations used by the comparison methods. Each library includes its license file.

| Directory | Upstream | Source revision | Used for |
|---|---|---|---|
| `deepSI` | https://github.com/MaartenSchoukens/deepSI | `109cf74f49a7b27539232c53981584befc1becc0` | SUBNET encoder, state transition and output networks |
| `DeepDynamicsModel` | https://github.com/linklab-uva/deep-dynamics | `88c856539c84fe7004b933d1cf03aad14ab69bd2` | Adaptive parameter encoder and vehicle prediction |

`identification.py:SUBNET` uses `deepSI.fit_systems.SS_encoder`. `models.py:load_authors_model` loads the Deep Dynamics network and physical model; architecture and parameter bounds come from the checkpoint configuration.

IAC CSVs and simulation archives in `datasets/` originate from the Deep Dynamics data used by the experiments. Prepared simulation windows can be regenerated with `datasets/prepare_tyre_sysid_dataset.py`.
