"""Allocation-free policy boundary; no native execution needed for these checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
from ps_flux_native95.contracts import INITIALIZATION_POLICY, geometry, rate_overrides, validate_options


def test_import_check_has_no_framework_side_effect():
    source = "from ps_flux_native95.cli import main; main(['check']); import sys; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, check=True, env=os.environ.copy())
    assert json.loads(result.stdout)["ema_decay"] is None


def test_rate_literals_and_no_phantom_prior():
    options = {"lr": .036, "d_lr_mult": .5000000000000001, "prior_lr_mult": None}
    result = rate_overrides(options)
    assert {key: value.hex() for key, value in result.items()} == {
        "lr": options["lr"].hex(), "d_lr_mult": options["d_lr_mult"].hex()}
    assert rate_overrides({}) == {}
    for value in (True, 0., float("nan"), float("inf"), -.001):
        with pytest.raises(ValueError):
            rate_overrides({"lr": value})
    with pytest.raises(ValueError, match="no particle prior"):
        rate_overrides({"prior_lr_mult": 1.})


def test_policy_refuses_before_paths():
    policy = {"pg_version": "0.9.5", "pg_source_profile": "release-0.9.5", "penalty_input_domain": "auto",
              "initialization_policy": INITIALIZATION_POLICY, "ema_decay": None}
    with patch.object(Path, "is_dir", side_effect=AssertionError("must reject before filesystem")):
        for changes in ({"ema_decay": .9}, {"pg_version": "0.9.4"}, {"lr": True}, {"prior_lr_mult": .1}):
            with pytest.raises(ValueError):
                validate_options(policy | changes)
    assert geometry({"height": 512, "width": 512, "text_tokens": 256})["image_tokens"] == 1024
    with pytest.raises(ValueError):
        geometry({"height": 511})
