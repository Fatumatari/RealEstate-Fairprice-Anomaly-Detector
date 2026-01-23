counties = {"Vihiga": "Mbale", "Uasin Gishu": "Eldoret", "Turkana": "Lodwar"}
for county in counties:
    print(county, counties[county], sep="- ")

counties =[{"name": "Vihiga", "hq": "Mbale", "population": 600000}, {"name": "Uasin Gishu", "hq": "Eldoret", "population": 1000000}, {"name": "Turkana", "hq": "Lodwar", "population": 450000}]
for c in counties:
    print(c['name'], c['hq'], c['population'], sep="-")