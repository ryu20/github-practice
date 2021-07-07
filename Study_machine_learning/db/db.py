import sqlite3


def create_table(dbname, create_table_sql):
    conn = sqlite3.connect(f'{dbname}.sqlite3')

    # カーソルを取得
    c = conn.cursor()

    # テーブル作成
    c.execute(create_table_sql)

    # コミット
    conn.commit()
    # コネクションをクローズ
    conn.close()
    return


if __name__ == '__main__':
    create_table('oanda_api', 'CREATE TABLE candles  (id INTEGER PRIMARY KEY AUTOINCREMENT, Time datetime,\
                    Volume int, Open real, High real, Low real, Close real)')
