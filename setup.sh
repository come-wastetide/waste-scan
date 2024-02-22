
sudo apt-get update
sudo apt-get upgrade -y

#Install the required packages

sudo apt-get install python3-pip python3-dev libpq-dev postgresql postgresql-contrib nginx curl -y

#Clone the repo

git clone https://github.com/come-wastetide/waste-scan.git

#Install the requirements in a virtual environment


sudo apt-get install python3-venv
python3 -m venv waste-scan-venv
source waste-scan-venv/bin/activate

cd waste-scan
pip install -r requirements.txt
 