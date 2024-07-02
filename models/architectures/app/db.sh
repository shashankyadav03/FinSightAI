#!/bin/sh
# brew services start postgresql

# Variables
DB_NAME=finsightdb
DB_USER=finuser
DB_PASS=finpass

# Create a new database
createdb $DB_NAME

# Create a new user
psql -c "CREATE USER $DB_USER WITH ENCRYPTED PASSWORD '$DB_PASS';"

# Grant all privileges on the database to the user
psql -c "GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;"
