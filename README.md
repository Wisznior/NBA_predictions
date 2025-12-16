# NBA_predictions

### Virtual environment
To create virtual environment:  
python3 -m venv .venv

To activate:  
source .venv/bin/activate

To deactivate:  
deactivate


### Managing dependencies
While being in the venv  

To install all required modules:  
pip install -r requirements.txt  

Before commiting run to save new dependencies:  
pip freeze > requirements.txt

### Starting server
run: python manage.py runserver

### Updating database structure changes
python manage.py makemigrations  
python manage.py migrate

### Managing Django Crontab
django-crontab is a wrapper around Linux cron.

Registering and updating list of jobs from settings.py CRONJOBS:
python manage.py crontab add

Listing os crontab jobs registered by django-crontab:
python manage.py crontab show

Removing all jobs registered by django-crontab:
python manage.py crontab remove

### Storing API Keys
Write them at the file ".env" in the project root

