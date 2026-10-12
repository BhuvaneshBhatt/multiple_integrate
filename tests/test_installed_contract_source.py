from pathlib import Path


def test_installed_contract_script_is_packaged_in_source_tree():
    script = Path(__file__).parents[1] / "tools" / "installed_wheel_contract.py"
    text = script.read_text()
    assert "asymptotic.multiseries" in text
    assert "properties.function_singularities" in text
    assert 'getattr(funcprops, "singularities"' in text
    assert "semialg.reduce_region_integral" in text
    assert "verify_regularization_result" in text
