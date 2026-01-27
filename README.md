# NBA_predictions

### Virtual environment
To create virtual environment:  
python -m venv venv

To activate:  
.\venv\Scripts\Activate.ps1 ( Windows )

source venv/bin/activate ( Linux )

To deactivate:  
deactivate

### Managing dependencies
While being in the venv  

To install all required modules:  
pip install -r requirements.txt  

Before commiting run to save new dependencies:  
pip freeze > requirements.txt

### Running script
./script.ps1 ( Windows )

chmod +x script.sh

./script.sh ( Linux )

If u have trouble running script in windows powershell, try saving the file with UTF-8 with BOM encoding

### Updating database structure changes
From nba_prophecy_website
python manage.py makemigrations  
python manage.py migrate

### Starting server
From nba_prophecy_website
python manage.py runserver

### Storing API Keys
Write them at the file ".env" in the project root
ODDS_API_KEY = *your_api_key*