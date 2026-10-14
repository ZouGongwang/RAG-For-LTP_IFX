import os, requests
import urllib3
urllib3.disable_warnings()    # hides the InsecureRequestWarning noise

BASE = "https://confluencewikiprod.intra.infineon.com"
H = {"Authorization": f"Bearer {os.environ['CONF_TOKEN']}"}

def t(path, **params):
    r = requests.get(BASE + path, headers=H, params=params, verify=False)
    print(path, params, "->", r.status_code, r.text[:200].replace("\n", " "))
    print()

t("/rest/api/user/current")
t("/rest/api/space/BEBSP")
t("/rest/api/content", spaceKey="BEBSP", type="page", limit=1)
t("/rest/api/content/search", cql="space=BEBSP and type=page", limit=1)