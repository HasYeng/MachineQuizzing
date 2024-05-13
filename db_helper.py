import mysql.connector  # Import MySQL connector for interacting with MySQL databases

def insert_or_update_progress(username, quiz_name, score):
    # Connect to the database
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="quiz_progress"
    )
    cursor = conn.cursor()

    # Check if entry exists for the given username and quiz_name
    cursor.execute(f"SELECT * FROM {quiz_name} WHERE username=%s", (username,))
    existing_entry = cursor.fetchone()

    if existing_entry:
        # Entry exists, update the values
        first_attempt = existing_entry[1]  # Get existing first_attempt score
        average = ((existing_entry[2] * existing_entry[4]) + score) / (existing_entry[4] + 1)
        last_attempt = existing_entry[5]
        num_attempts = existing_entry[4] + 1
        current_attempt = score

        # Update the row
        cursor.execute(f"UPDATE {quiz_name} SET first_attempt=%s, average=%s, last_attempt=%s, num_attempts=%s,"
                       f" current_attempt=%s WHERE username=%s",
                       (first_attempt, average, last_attempt, num_attempts, current_attempt, username))
    else:
        # Entry does not exist, insert new row
        first_attempt = score
        average = score
        last_attempt = score
        num_attempts = 1
        current_attempt = score

        # Insert new row
        cursor.execute(
            f"INSERT INTO {quiz_name} (username, first_attempt, average, last_attempt, num_attempts, current_attempt) "
            f"VALUES (%s, %s, %s, %s, %s, %s)",
            (username, first_attempt, average, last_attempt, num_attempts, current_attempt))

    # Commit changes and close connection
    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Example usage
    insert_or_update_progress('user1237', 'general_ml_quiz1', 95)
    insert_or_update_progress('user1245', 'general_ml_quiz1', 50)

