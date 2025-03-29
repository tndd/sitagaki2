from fixture.infra.model import DataSchemaImpl, factory_data_schema_impl


def test_factory_data_schema_impl():
    dsi: DataSchemaImpl = factory_data_schema_impl()
    assert isinstance(dsi, DataSchemaImpl)
    assert dsi.df.index.name == "Date"
