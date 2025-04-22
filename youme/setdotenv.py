import os.path

from dotenv import dotenv_values, set_key

DEFAULT_ENV_FILE = ".env.template"
SECRETS_ENV_FILE = ".env.secrets"


if os.path.exists(DEFAULT_ENV_FILE) == False:
    print("Default env file does not exist")
    exit(1)

# Load the default key-value pairs from the .env.template file
env_values = dotenv_values(DEFAULT_ENV_FILE)

# Load the key-value pairs from the .env.template file to be updated
update_env_values = dotenv_values(SECRETS_ENV_FILE)

# Update the default values
env_values.update(update_env_values)

# Write the updated key-value pairs to the .env file
for key, value in env_values.items():
    set_key(".env", key, value)
