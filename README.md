# NBA_predictions

### Virtual environment
To create virtual environment:  
python3 -m venv venv

To activate:  
.\venv\Scripts\Activate.ps1

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

### Storing API Keys
Write them at the file ".env" in the project root

