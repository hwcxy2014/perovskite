import pandas
import sqlalchemy

__author__ = 'jialeiwang'

sql_engine = sqlalchemy.create_engine('mysql+mysqldb://jialeiwang:wangjialei123@work.cxcjqzn7ydtp.us-east-1.rds.amazonaws.com/cold_start')

def write_to_table(table_name, values):
    """
    :param table_name: string; name of the table to write to.
    :param values: list of tuples; each tuple corresponds to one row in the table
    :return:
    """
    sql_table = sqlalchemy.Table(table_name, sqlalchemy.MetaData(), autoload=True, autoload_with=sql_engine)
    conn = sql_engine.connect()
    conn.execute(sql_table.insert().values(values))


def write_array_to_table(table_name, array):
    """
    :param table_name: string; name of the table to write to.
    :param array: numpy.array
    :return:
    """
    df = pandas.DataFrame()
    for i in range(len(array)):
        df['p{0}'.format(i)] = [array[i]]
    df.to_sql(table_name, sql_engine, if_exists='append', index=False)
