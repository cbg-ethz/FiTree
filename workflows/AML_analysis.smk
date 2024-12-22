######### Imports #########
import fitree
import pymc as pm
import jax
import arviz as az
import os
import numpy as np
import pytensor

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)


######### Setup #########

# Parameters
N_TUNE: list[int] = [500, 1000]
N_DRAW: list[int] = [500, 1000]
N_CHAINS: int = 24
RHS_LOCAL_SCALE: list[float] = [0.5, 0.75, 1.0]
RHS_S2: list[float] = [0.25, 0.5, 1.0]
SEED: int = 2025
working_dir: str = os.getcwd()
script_dir: str = "/cluster/home/luox/FiTree/workflows/scripts"
base_temp_dir: str = "/scratch/{}/pytensor_temp".format(
    os.environ.get("USER", "unknown_user")
)

# Ensure base directory for PyTensor temp files exists
os.makedirs(base_temp_dir, exist_ok=True)

######### Workflow #########

# Generate RHS pairs
RHS_pairs = expand(
    "{RHS_local_scale}_{RHS_s2}",
    zip,
    RHS_local_scale=RHS_LOCAL_SCALE,
    RHS_s2=RHS_S2,
)


rule all:
    input:
        expand(
            "results/AML_no_mask_tune{n_tune}_draw{n_draw}_RHS{rhs}.nc",
            n_tune=N_TUNE,
            n_draw=N_DRAW,
            rhs=RHS_pairs,
        ),


rule run_fitree_no_mask:
    output:
        "results/AML_no_mask_tune{n_tune}_draw{n_draw}_RHS{RHS_local_scale}_{RHS_s2}.nc",
    threads: N_CHAINS
    resources:
        runtime=10000,
        tasks=1,
        nodes=1,
    shell:
        """
        sleep 60

        # Create a unique temporary directory for this job
        PYTENSOR_TEMP_DIR={base_temp_dir}/{wildcards.n_tune}_{wildcards.n_draw}_{wildcards.RHS_local_scale}_{wildcards.RHS_s2}
        mkdir -p $PYTENSOR_TEMP_DIR
        export PYTENSOR_FLAGS=base_compiledir=$PYTENSOR_TEMP_DIR

        # Run the fitree script
        cd {script_dir} && \
        python run_fitree_AML.py \
            --n_tune {wildcards.n_tune} \
            --n_samples {wildcards.n_draw} \
            --n_chains {threads} \
            --local_scale {wildcards.RHS_local_scale} \
            --s2 {wildcards.RHS_s2} \
            --workdir {working_dir} \
            --seed {SEED} \
            --mask False

        # Cleanup the temporary directory
        rm -rf $PYTENSOR_TEMP_DIR
        """
