from utils.dict_merge import deep_merge, unflatten


class TestUnflatten:
    def test_single_key(self):
        assert unflatten("a", 1) == {"a": 1}

    def test_nested_path(self):
        assert unflatten("a.b.c", 5) == {"a": {"b": {"c": 5}}}

    def test_dict_value(self):
        assert unflatten("arch", {"outer": 128}) == {"arch": {"outer": 128}}


class TestDeepMerge:
    def test_disjoint_keys(self):
        assert deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_scalar_replace(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_nested_merge_preserves_siblings(self):
        base = {"arch": {"obs": {"n": 5}, "outer": 64}}
        overlay = {"arch": {"outer": 128}}
        result = deep_merge(base, overlay)
        assert result == {"arch": {"obs": {"n": 5}, "outer": 128}}

    def test_compound_axis_preserves_nested(self):
        """Compound dict value at 'arch' shouldn't clobber arch.obs."""
        base = {"arch": {"obs": {"n": 5}}}
        overlay = unflatten("arch", {"outer": 128, "inner": 160})
        result = deep_merge(base, overlay)
        assert result == {"arch": {"obs": {"n": 5}, "outer": 128, "inner": 160}}

    def test_two_axes_shallow_then_deep(self):
        """The tricky case: compound arch axis + scalar arch.obs.n axis."""
        base = {"arch": {"obs": {"n": 5}, "outer": 64}}
        step1 = deep_merge(base, unflatten("arch", {"outer": 128, "inner": 160}))
        step2 = deep_merge(step1, unflatten("arch.obs.n", 20))
        assert step2 == {"arch": {"obs": {"n": 20}, "outer": 128, "inner": 160}}

    def test_no_mutation_of_inputs(self):
        base = {"a": {"b": 1}}
        overlay = {"a": {"c": 2}}
        result = deep_merge(base, overlay)
        assert result == {"a": {"b": 1, "c": 2}}
        assert base == {"a": {"b": 1}}
        assert overlay == {"a": {"c": 2}}

    def test_overlay_replaces_scalar_with_dict(self):
        assert deep_merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}

    def test_overlay_replaces_dict_with_scalar(self):
        assert deep_merge({"a": {"b": 1}}, {"a": 99}) == {"a": 99}

    def test_empty_overlay(self):
        assert deep_merge({"a": 1}, {}) == {"a": 1}

    def test_empty_base(self):
        assert deep_merge({}, {"a": 1}) == {"a": 1}
