import pandas as pd
import re
from collections import defaultdict, Counter

name_canonicals = {
    # United States
    "usps": "USPS",
    "ups": "UPS",
    "fedex": "FedEx",
    "amazon": "Amazon Logistics",
    "amazon logistics": "Amazon Logistics",
    "ontrac": "OnTrac",
    "lasership": "LaserShip",

    # United Kingdom
    "evri": "EVRi",
    "dpd uk": "DPD UK",
    "royal mail": "Royal Mail",
    "yodel": "Yodel",
    "parcelforce worldwide": "Parcelforce Worldwide",
    "amazon logistics uk": "Amazon Logistics UK",

    # Canada
    "canada post": "Canada Post",
    "purolator": "Purolator",
    "canpar express": "Canpar Express",
    "intelcom": "Intelcom",
    "nationex": "Nationex",

    # Europe
    "dpd": "DPD",
    "dhl": "DHL",
    "db schenker": "DB Schenker",
    "gls": "GLS",
    "la poste": "La Poste",
    "tnt express": "TNT Express",
    "chronopost": "Chronopost",
    "postnl": "PostNL",
    "correos": "Correos",
    "bpost": "Bpost",
    "poste italiane": "Poste Italiane",
    "česká pošta": "Česká pošta",
    "swiss post": "Swiss Post",

    # Australia & New Zealand
    "australia post": "Australia Post",
    "startrack": "StarTrack",
    "couriersplease": "CouriersPlease",
    "sendle": "Sendle",
    "nz post": "NZ Post",

    # China
    "china post": "China Post",
    "sf express": "SF Express",
    "yto express": "YTO Express",
    "zto express": "ZTO Express",
    "jd logistics": "JD Logistics",
    "cainiao": "Cainiao",

    # Japan
    "japan post": "Japan Post",
    "yamato transport": "Yamato Transport",
    "sagawa express": "Sagawa Express",

    # South Korea
    "korea post": "Korea Post",
    "cj logistics": "CJ Logistics",
    "hanjin express": "Hanjin Express",
    "lotte global logistics": "Lotte Global Logistics",

    # India
    "india post": "India Post",
    "delhivery": "Delhivery",
    "blue dart": "Blue Dart",
    "ecom express": "Ecom Express",
    "xpressbees": "XpressBees",
    "dtdc": "DTDC",

    # Philippines
    "lbc express": "LBC Express",
    "j&t express philippines": "J&T Express Philippines",
    "ninja van philippines": "Ninja Van Philippines",
    "2go express": "2GO Express",
    "entrego": "Entrego",
    "xde logistics": "XDE Logistics",

    # Nigeria
    "abc transport": "ABC Transport",
    "red star express": "Red Star Express",
    "tranex": "Tranex",
    "ems nigeria": "EMS Nigeria",
    "max.ng": "Max.ng",
    "kwik delivery": "Kwik Delivery",
    "courierplus": "CourierPlus",
    "gig logistics": "GIG Logistics",

    # Thailand
    "thailand post": "Thailand Post",
    "kerry express": "Kerry Express",
    "flash express": "Flash Express",
    "scg express": "SCG Express",
    "j&t express thailand": "J&T Express Thailand",
    
    # Brazil
    "correios": "Correios",
    "total express": "Total Express",
    "jadlog": "Jadlog",
    "loggi": "Loggi",
    "mercado livre": "Mercado Livre",
    "sequoia logistica e transportes": "Sequoia Logística e Transportes",
    
    # Mexico
    "correos de méxico": "Correos de México",
    "correos de mexico": "Correos de México",
    "estafeta": "Estafeta",
    "paquetexpress": "Paquetexpress",
    "redpack": "Redpack",
    "grupo ampm": "Grupo ampm",
    "99 minutos": "99 Minutos",

    # Middle East & Africa
    "aramex": "Aramex",
    "saudi post": "Saudi Post",
    "egypt post": "Egypt Post",
    "posta uganda": "Posta Uganda"
}

companies_by_country = {
    "United States": ["USPS", "UPS", "FedEx", "Amazon Logistics", "OnTrac", "LaserShip"],
    "United Kingdom": ["EVRi", "DPD UK", "Royal Mail", "Yodel", "Parcelforce Worldwide", "Amazon Logistics UK"],
    "Canada": ["Canada Post", "Purolator", "Canpar Express", "Intelcom", "Nationex"],
    "Europe": ["DPD", "DHL", "DB Schenker", "GLS", "La Poste", "TNT Express", "Chronopost", "PostNL", "Correos", "Bpost", "Poste Italiane", "Česká pošta", "Swiss Post"],
    "Australia & New Zealand": ["Australia Post", "StarTrack", "CouriersPlease", "Sendle", "NZ Post"],
    "China": ["China Post", "SF Express", "YTO Express", "ZTO Express", "JD Logistics", "Cainiao"],
    "Japan": ["Japan Post", "Yamato Transport", "Sagawa Express"],
    "South Korea": ["Korea Post", "CJ Logistics", "Hanjin Express", "Lotte Global Logistics"],
    "India": ["India Post", "Delhivery", "Blue Dart", "Ecom Express", "XpressBees", "DTDC"],
    "Philippines": ["LBC Express", "J&T Express Philippines", "Ninja Van Philippines", "2GO Express", "Entrego", "XDE Logistics"],
    "Nigeria": ["ABC Transport", "Red Star Express", "Tranex", "EMS Nigeria", "Max.ng", "Kwik Delivery", "CourierPlus", "GIG Logistics"],
    "Thailand": ["Thailand Post", "Kerry Express", "Flash Express", "SCG Express", "J&T Express Thailand"],
    "Brazil": ["Correios", "Total Express", "Jadlog", "Loggi", "Mercado Livre", "Sequoia Logística e Transportes"],
    "Mexico": ["Correos de México", "Estafeta", "Paquetexpress", "Redpack", "Grupo ampm", "99 Minutos"],
    "Middle East & Africa": ["Aramex", "Saudi Post", "Egypt Post", "Posta Uganda"]
}


try:
    # Add path to CSV
    df = pd.read_csv("data.csv")
    text_col = "Extracted Text"

    pattern = r'\b(?:' + '|'.join(re.escape(k) for k in name_canonicals.keys()) + r')\b'
    compiled = re.compile(pattern, flags=re.IGNORECASE)

    counter = Counter()
    for text in df[text_col].dropna():
        for match in compiled.findall(text):
            canonical = name_canonicals[match.lower()]
            counter[canonical] += 1

    with open("logistics_counts_output.txt", "w") as f:
        f.write("Match frequency by country/region and delivery company:\n\n")
        for country, companies in companies_by_country.items():
            f.write(f"=== {country} ===\n")
            for name in companies:
                if counter[name]:
                    f.write(f"{name}: {counter[name]}\n")
            f.write("\n")

except FileNotFoundError:
    print("Error: The file was not found. Please ensure the file is in the same directory as the script.")