import psycopg2


def connect():
    conn = psycopg2.connect("dbname=vectordb user=postgres password=postgres")
    cursor = conn.cursor()

    return conn, cursor


def insert_into_db(content, embedding, cursor):
    cursor.execute(
        "INSERT INTO items (content, embedding) VALUES (%s, %s)", (content, embedding)
    )


def find_similar_vectors(embed, cursor):
    cursor.execute(
        """
SELECT id, content
FROM items
ORDER BY embedding <-> %s::vector
LIMIT 5
""",
        (embed,),
    )

    results = cursor.fetchall()

    for row in results:
        print(row)
