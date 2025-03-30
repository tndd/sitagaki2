from fixture.infra.model import DatasetImpl, factory_dataset_impl


def test_factory_dataset():
    dsi: DatasetImpl = factory_dataset_impl()
    assert isinstance(dsi, DatasetImpl)
    assert dsi.df.index.name == "Date"
