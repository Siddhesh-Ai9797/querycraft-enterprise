import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
load_dotenv()

def get_create_table_string(engine, table_name: str) -> str:
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name, schema="public")
    column_definitions = []
    for col in columns:
        col_type = str(col["type"])
        col_type = col_type.replace("VARCHAR(length=", "VARCHAR(")
        col_type = col_type.replace("NUMERIC(precision=10, scale=2)", "NUMERIC(10,2)")
        col_def = f"{col['name']} {col_type}"
        if not col.get("nullable", True):
            col_def += " NOT NULL"
        column_definitions.append(col_def)
    columns_str = ", ".join(column_definitions)
    return f"CREATE TABLE {table_name} ({columns_str})"

def get_schema_for_tables(engine, table_names: list) -> str:
    create_statements = []
    for table_name in table_names:
        create_statements.append(get_create_table_string(engine, table_name))
    return "\n".join(create_statements)

def get_full_schema(engine) -> dict:
    inspector = inspect(engine)
    all_tables = inspector.get_table_names(schema="public")
    schema_dict = {}
    for table_name in sorted(all_tables):
        schema_dict[table_name] = get_create_table_string(engine, table_name)
    return schema_dict

if __name__ == "__main__":
    engine = create_engine(os.getenv("DATABASE_URL"))
    print("=== Single Table ===")
    print(get_create_table_string(engine, "artist"))
    print("\n=== Full Schema Dictionary ===")
    full = get_full_schema(engine)
    for table, ddl in full.items():
        print(f"  {table}: {len(ddl)} chars")