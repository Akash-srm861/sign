from database import engine
from sqlalchemy import text

# Add password column to users table
with engine.connect() as conn:
    try:
        conn.execute(text('ALTER TABLE users ADD COLUMN password VARCHAR(255)'))
        conn.commit()
        print('✓ Password column added to users table')
    except Exception as e:
        if 'already exists' in str(e):
            print('✓ Password column already exists')
        else:
            print(f'Error: {e}')
