############################################################################

# Dibuat oleh Metalearning Ibrahimy
# UNIB - Universitas Ibrahimy (Indonesia)
# email:  taufiqurrahman.info@gmail.com
# IpyBibX - A Bibliometric and Scientometric Library Powered with Artificial Intelligence Tools

# Citation: 
# RAHMAN, T. (2023). Project: IpyBibX, File: ipbibx.py, GitHub repository: <https://github.com/metalearning-ibrahimy/IpyBibX>

############################################################################

# Required Libraries
import chardet
import networkx as nx             
import numpy as np   
import openai       
import os       
import pandas as pd               
import plotly.graph_objects as go
import plotly.subplots as ps      
import plotly.io as pio           
import re                         
import squarify                  
import unicodedata                
import textwrap

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
from . import stws

from bertopic import BERTopic                               
from collections import Counter
from difflib import SequenceMatcher
from matplotlib import pyplot as plt                       
plt.style.use('bmh')
#from scipy.spatial import ConvexHull   
from sentence_transformers import SentenceTransformer                    
from sklearn.cluster import KMeans                          
from sklearn.decomposition import TruncatedSVD as tsvd      
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity  
from summarizer import Summarizer
from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from umap import UMAP  
from wordcloud import WordCloud                           

############################################################################

# ipbx Class
class ipbx_probe():
    def __init__(self, file_bib, db = 'scopus', del_duplicated = True):
        self.institution_names =  [ 
                                    'acad', 'academy', 'akad', 'aachen', 'assoc', 'cambridge', 'ctr', 'cefet', 'center', 'centre', 'chuo kikuu', 
                                    'cient', 'coll', 'college', 'colegio', 'conservatory', 'dept', 'egyetemi', 'escola', 'education', 'escuela', 
                                    'eyunivesithi', 'fac', 'faculdade', 'facultad', 'fakultet', 'fakultät', 'fdn', 'fundacion', 'foundation', 
                                    'gradevinski', 'higher', 'hsch', 'hochschule', 'hosp', 'hgsk', 'hogeschool',  'háskóli', 'högskola', 'ibmec', 
                                    'inivèsite', 'ist', 'istituto', 'imd', 'institutional', 'int', 'inst',  'institut', 'institute', 
                                    'institute of technology',  'inyuvesi', 'iskola', 'iunivesite', 'jaamacad', "jami'a",  'kolej', 'koulu', 
                                    'kulanui', 'lab.', 'lab', 'labs', 'laborat', 'learning', 'mahadum', 'med', 'medicine', 'medical', 'observatory', 
                                    'oilthigh', 'okulu', 'ollscoile', 'oniversite', 'politecnico', 'polytechnic', 'prifysgol', 'rech', 'recherche', 
                                    'research', 'sch', 'school', 'schule', 'scuola', 'seminary', 'skola', 'supérieur', 'sveučilište', 'szkoła', 
                                    'tech', 'technical', 'technische', 'technique', 'technological', 'uff', 'uned', 'unibersidad', 'unibertsitatea', 
                                    'univ', 'universidad', 'universidade', 'universitas', 'universitat', 'universitate', 'universitato', 
                                    'universite', 'universiteit', 'universitet', 'universitetas', 'universiti', 'university', 'università', 
                                    'universität', 'université', 'universitāte', 'univerza', 'univerzita','univerzitet', 'univesithi', 'uniwersytet', 
                                    'vniuersitatis', 'whare wananga', 'yliopisto','yunifasiti', 'yunivesite', 'yunivhesiti', 'zanko', 'école', 
                                    'ülikool', 'üniversite','πανεπιστήμιο', 'σχολείο', 'универзитет', 'университет', 'універсітэт', 'школа'
                                  ]

        self.language_names  =    { 
                                    'afr': 'Afrikaans', 'alb': 'Albanian','amh': 'Amharic', 'ara': 'Arabic', 'arm': 'Armenian', 
                                    'aze': 'Azerbaijani', 'bos': 'Bosnian', 'bul': 'Bulgarian', 'cat': 'Catalan', 'chi': 'Chinese', 
                                    'cze': 'Czech', 'dan': 'Danish', 'dut': 'Dutch', 'eng': 'English', 'epo': 'Esperanto', 
                                    'est': 'Estonian', 'fin': 'Finnish', 'fre': 'French', 'geo': 'Georgian', 'ger': 'German', 
                                    'gla': 'Scottish Gaelic', 'gre': 'Greek, Modern', 'heb': 'Hebrew', 'hin': 'Hindi', 
                                    'hrv': 'Croatian', 'hun': 'Hungarian', 'ice': 'Icelandic', 'ind': 'Indonesian', 'ita': 'Italian', 
                                    'jpn': 'Japanese', 'kin': 'Kinyarwanda', 'kor': 'Korean', 'lat': 'Latin', 'lav': 'Latvian', 
                                    'lit': 'Lithuanian', 'mac': 'Macedonian', 'mal': 'Malayalam', 'mao': 'Maori', 'may': 'Malay', 
                                    'mul': 'Multiple languages', 'nor': 'Norwegian', 'per': 'Persian, Iranian', 'pol': 'Polish', 
                                    'por': 'Portuguese', 'pus': 'Pushto', 'rum': 'Romanian, Rumanian, Moldovan', 'rus': 'Russian', 
                                    'san': 'Sanskrit', 'slo': 'Slovak', 'slv': 'Slovenian', 'spa': 'Spanish', 'srp': 'Serbian', 
                                    'swe': 'Swedish', 'tha': 'Thai', 'tur': 'Turkish', 'ukr': 'Ukrainian', 'und': 'Undetermined', 
                                    'vie': 'Vietnamese', 'wel': 'Welsh'
                                  }
        self.country_names =      [
                                   'Afghanistan', 'Albania', 'Algeria', 'American Samoa', 'Andorra', 'Angola', 'Anguilla', 
                                   'Antarctica', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Aruba', 'Australia', 'Austria', 
                                   'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 
                                   'Benin', 'Bermuda', 'Bhutan', 'Bolivia', 'Bonaire, Sint Eustatius and Saba', 
                                   'Bosnia and Herzegovina', 'Botswana', 'Bouvet Island', 'Brazil', 'British Indian Ocean Territory', 
                                   'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 
                                   'Canada', 'Cayman Islands', 'Central African Republic', 'Chad', 'Chile', 'China', 
                                   'Christmas Island', 'Cocos Islands', 'Colombia', 'Comoros', 'Democratic Republic of the Congo', 
                                   'Congo', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Curacao', 'Cyprus', 'Czechia', 
                                   "Côte d'Ivoire", 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 
                                   'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 
                                   'Falkland Islands', 'Faroe Islands', 'Fiji', 'Finland', 'France', 'French Guiana', 
                                   'French Polynesia', 'French Southern Territories', 'Gabon', 'Gambia', 'Georgia', 'Germany', 
                                   'Ghana', 'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guadeloupe', 'Guam', 'Guatemala', 
                                   'Guernsey', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Heard Island and McDonald Islands', 
                                   'Holy See', 'Honduras', 'Hong Kong', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 
                                   'Ireland', 'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Kazakhstan', 
                                   'Kenya', 'Kiribati', 'North Korea', 'South Korea', 'Kuwait', 'Kyrgyzstan', 
                                   "Lao People's Democratic Republic", 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 
                                   'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macao', 'Madagascar', 'Malawi', 'Malaysia', 
                                   'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Martinique', 'Mauritania', 'Mauritius', 
                                   'Mayotte', 'Mexico', 'Micronesia', 'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Montserrat', 
                                   'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Caledonia', 
                                   'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'Norfolk Island', 
                                   'Northern Mariana Islands', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 
                                   'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Pitcairn', 'Poland', 'Portugal', 
                                   'Puerto Rico', 'Qatar', 'Republic of North Macedonia', 'Romania', 'Russian Federation', 'Rwanda', 
                                   'Réunion', 'Saint Barthelemy', 'Saint Helena, Ascension and Tristan da Cunha', 
                                   'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin', 'Saint Pierre and Miquelon', 
                                   'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 
                                   'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Sint Maarten', 'Slovakia', 
                                   'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 
                                   'South Georgia and the South Sandwich Islands', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 
                                   'Suriname', 'Svalbard and Jan Mayen', 'Sweden', 'Switzerland', 'Syrian Arab Republic', 'Taiwan', 
                                   'Tajikistan', 'Tanzania', 'Thailand', 'Timor-Leste', 'Togo', 'Tokelau', 'Tonga', 
                                   'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu', 
                                   'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom', 
                                   'United States Minor Outlying Islands', 'United States of America', 'Uruguay', 'Uzbekistan', 
                                   'Vanuatu', 'Venezuela', 'Viet Nam', 'Virgin Islands (British)', 'Virgin Islands (U.S.)', 
                                   'Wallis and Futuna', 'Western Sahara', 'Yemen', 'Zambia', 'Zimbabwe', 'Aland Islands'
                                  ] 
        self.country_alpha_2 =    [
                                   'AF', 'AL', 'DZ', 'AS', 'AD', 'AO', 'AI', 'AQ', 'AG', 'AR', 'AM', 'AW', 'AU', 'AT', 'AZ', 'BS', 
                                   'BH', 'BD', 'BB', 'BY', 'BE', 'BZ', 'BJ', 'BM', 'BT', 'BO', 'BQ', 'BA', 'BW', 'BV', 'BR', 'IO', 
                                   'BN', 'BG', 'BF', 'BI', 'CV', 'KH', 'CM', 'CA', 'KY', 'CF', 'TD', 'CL', 'CN', 'CX', 'CC', 'CO', 
                                   'KM', 'CD', 'CG', 'CK', 'CR', 'HR', 'CU', 'CW', 'CY', 'CZ', 'CI', 'DK', 'DJ', 'DM', 'DO', 'EC', 
                                   'EG', 'SV', 'GQ', 'ER', 'EE', 'SZ', 'ET', 'FK', 'FO', 'FJ', 'FI', 'FR', 'GF', 'PF', 'TF', 'GA', 
                                   'GM', 'GE', 'DE', 'GH', 'GI', 'GR', 'GL', 'GD', 'GP', 'GU', 'GT', 'GG', 'GN', 'GW', 'GY', 'HT', 
                                   'HM', 'VA', 'HN', 'HK', 'HU', 'IS', 'IN', 'ID', 'IR', 'IQ', 'IE', 'IM', 'IL', 'IT', 'JM', 'JP', 
                                   'JE', 'JO', 'KZ', 'KE', 'KI', 'KP', 'KR', 'KW', 'KG', 'LA', 'LV', 'LB', 'LS', 'LR', 'LY', 'LI', 
                                   'LT', 'LU', 'MO', 'MG', 'MW', 'MY', 'MV', 'ML', 'MT', 'MH', 'MQ', 'MR', 'MU', 'YT', 'MX', 'FM', 
                                   'MD', 'MC', 'MN', 'ME', 'MS', 'MA', 'MZ', 'MM', 'NA', 'NR', 'NP', 'NL', 'NC', 'NZ', 'NI', 'NE', 
                                   'NG', 'NU', 'NF', 'MP', 'NO', 'OM', 'PK', 'PW', 'PS', 'PA', 'PG', 'PY', 'PE', 'PH', 'PN', 'PL', 
                                   'PT', 'PR', 'QA', 'MK', 'RO', 'RU', 'RW', 'RE', 'BL', 'SH', 'KN', 'LC', 'MF', 'PM', 'VC', 'WS', 
                                   'SM', 'ST', 'SA', 'SN', 'RS', 'SC', 'SL', 'SG', 'SX', 'SK', 'SI', 'SB', 'SO', 'ZA', 'GS', 'SS', 
                                   'ES', 'LK', 'SD', 'SR', 'SJ', 'SE', 'CH', 'SY', 'TW', 'TJ', 'TZ', 'TH', 'TL', 'TG', 'TK', 'TO', 
                                   'TT', 'TN', 'TR', 'TM', 'TC', 'TV', 'UG', 'UA', 'AE', 'GB', 'UM', 'US', 'UY', 'UZ', 'VU', 'VE', 
                                   'VN', 'VG', 'VI', 'WF', 'EH', 'YE', 'ZM', 'ZW', 'AX'
                                  ]
        self.country_alpha_3 =    [
                                   'AFG', 'ALB', 'DZA', 'ASM', 'AND', 'AGO', 'AIA', 'ATA', 'ATG', 'ARG', 'ARM', 'ABW', 'AUS', 'AUT', 
                                   'AZE', 'BHS', 'BHR', 'BGD', 'BRB', 'BLR', 'BEL', 'BLZ', 'BEN', 'BMU', 'BTN', 'BOL', 'BES', 'BIH', 
                                   'BWA', 'BVT', 'BRA', 'IOT', 'BRN', 'BGR', 'BFA', 'BDI', 'CPV', 'KHM', 'CMR', 'CAN', 'CYM', 'CAF', 
                                   'TCD', 'CHL', 'CHN', 'CXR', 'CCK', 'COL', 'COM', 'COD', 'COG', 'COK', 'CRI', 'HRV', 'CUB', 'CUW', 
                                   'CYP', 'CZE', 'CIV', 'DNK', 'DJI', 'DMA', 'DOM', 'ECU', 'EGY', 'SLV', 'GNQ', 'ERI', 'EST', 'SWZ', 
                                   'ETH', 'FLK', 'FRO', 'FJI', 'FIN', 'FRA', 'GUF', 'PYF', 'ATF', 'GAB', 'GMB', 'GEO', 'DEU', 'GHA', 
                                   'GIB', 'GRC', 'GRL', 'GRD', 'GLP', 'GUM', 'GTM', 'GGY', 'GIN', 'GNB', 'GUY', 'HTI', 'HMD', 'VAT', 
                                   'HND', 'HKG', 'HUN', 'ISL', 'IND', 'IDN', 'IRN', 'IRQ', 'IRL', 'IMN', 'ISR', 'ITA', 'JAM', 'JPN', 
                                   'JEY', 'JOR', 'KAZ', 'KEN', 'KIR', 'PRK', 'KOR', 'KWT', 'KGZ', 'LAO', 'LVA', 'LBN', 'LSO', 'LBR', 
                                   'LBY', 'LIE', 'LTU', 'LUX', 'MAC', 'MDG', 'MWI', 'MYS', 'MDV', 'MLI', 'MLT', 'MHL', 'MTQ', 'MRT', 
                                   'MUS', 'MYT', 'MEX', 'FSM', 'MDA', 'MCO', 'MNG', 'MNE', 'MSR', 'MAR', 'MOZ', 'MMR', 'NAM', 'NRU', 
                                   'NPL', 'NLD', 'NCL', 'NZL', 'NIC', 'NER', 'NGA', 'NIU', 'NFK', 'MNP', 'NOR', 'OMN', 'PAK', 'PLW', 
                                   'PSE', 'PAN', 'PNG', 'PRY', 'PER', 'PHL', 'PCN', 'POL', 'PRT', 'PRI', 'QAT', 'MKD', 'ROU', 'RUS', 
                                   'RWA', 'REU', 'BLM', 'SHN', 'KNA', 'LCA', 'MAF', 'SPM', 'VCT', 'WSM', 'SMR', 'STP', 'SAU', 'SEN', 
                                   'SRB', 'SYC', 'SLE', 'SGP', 'SXM', 'SVK', 'SVN', 'SLB', 'SOM', 'ZAF', 'SGS', 'SSD', 'ESP', 'LKA', 
                                   'SDN', 'SUR', 'SJM', 'SWE', 'CHE', 'SYR', 'TWN', 'TJK', 'TZA', 'THA', 'TLS', 'TGO', 'TKL', 'TON', 
                                   'TTO', 'TUN', 'TUR', 'TKM', 'TCA', 'TUV', 'UGA', 'UKR', 'ARE', 'GBR', 'UMI', 'USA', 'URY', 'UZB', 
                                   'VUT', 'VEN', 'VNM', 'VGB', 'VIR', 'WLF', 'ESH', 'YEM', 'ZMB', 'ZWE', 'ALA'
                                  ]
        self.country_numeric =    [
                                    4, 8, 12, 16, 20, 24, 660, 10, 28, 32, 51, 533, 36, 40, 31, 44, 48, 50, 52, 112, 56, 84, 204, 60, 
                                   64, 68, 535, 70, 72, 74, 76, 86, 96, 100, 854, 108, 132, 116, 120, 124, 136, 140, 148, 152, 156, 
                                   162, 166, 170, 174, 180, 178, 184, 188, 191, 192, 531, 196, 203, 384, 208, 262, 212, 214, 218, 818, 
                                   222, 226, 232, 233, 748, 231, 238, 234, 242, 246, 250, 254, 258, 260, 266, 270, 268, 276, 288, 292, 
                                   300, 304, 308, 312, 316, 320, 831, 324, 624, 328, 332, 334, 336, 340, 344, 348, 352, 356, 360, 364, 
                                   368, 372, 833, 376, 380, 388, 392, 832, 400, 398, 404, 296, 408, 410, 414, 417, 418, 428, 422, 426, 
                                   430, 434, 438, 440, 442, 446, 450, 454, 458, 462, 466, 470, 584, 474, 478, 480, 175, 484, 583, 498, 
                                   492, 496, 499, 500, 504, 508, 104, 516, 520, 524, 528, 540, 554, 558, 562, 566, 570, 574, 580, 578, 
                                   512, 586, 585, 275, 591, 598, 600, 604, 608, 612, 616, 620, 630, 634, 807, 642, 643, 646, 638, 652, 
                                   654, 659, 662, 663, 666, 670, 882, 674, 678, 682, 686, 688, 690, 694, 702, 534, 703, 705, 90, 706, 
                                   710, 239, 728, 724, 144, 729, 740, 744, 752, 756, 760, 158, 762, 834, 764, 626, 768, 772, 776, 780, 
                                   788, 792, 795, 796, 798, 800, 804, 784, 826, 581, 840, 858, 860, 548, 862, 704, 92, 850, 876, 732, 
                                   887, 894, 716, 248
                                  ] 
        self.country_lat_long =   [
                                   (33.93911, 67.709953),    (41.153332, 20.168331),    (28.033886, 1.659626),   (-14.270972, -170.132217), 
                                   (42.546245, 1.601554),    (-11.202692, 17.873887),   (18.220554, -63.068615), (-75.250973, -0.071389), 
                                   (17.060816, -61.796428),  (-38.416097, -63.616672),  (40.069099, 45.038189),  (12.52111, -69.968338), 
                                   (-25.274398, 133.775136), (47.516231, 14.550072),    (40.143105, 47.576927),  (25.03428, -77.39628), 
                                   (25.930414, 50.637772),   (23.684994, 90.356331),    (13.193887, -59.543198), (53.709807, 27.953389), 
                                   (50.503887, 4.469936),    (17.189877, -88.49765),    (9.30769, 2.315834),     (32.321384, -64.75737), 
                                   (27.514162, 90.433601),   (-16.290154, -63.588653),  (12.15, -68.26667),      (43.915886, 17.679076), 
                                   (-22.328474, 24.684866),  (-54.423199, 3.413194),    (-14.235004, -51.92528), (-6.343194, 71.876519), 
                                   (4.535277, 114.727669),   (42.733883, 25.48583),     (12.238333, -1.561593),  (-3.373056, 29.918886), 
                                   (16.002082, -24.013197),  (12.565679, 104.990963),   (7.369722, 12.354722),   (56.130366, -106.346771), 
                                   (19.513469, -80.566956),  (6.611111, 20.939444),     (15.454166, 18.732207),  (-35.675147, -71.542969), 
                                   (35.86166, 104.195397),   (-10.447525, 105.690449),  (-12.164165, 96.870956), (4.570868, -74.297333), 
                                   (-11.875001, 43.872219),  (-4.038333, 21.758664),    (-0.228021, 15.827659),  (-21.236736, -159.777671), 
                                   (9.748917, -83.753428),   (45.1, 15.2),              (21.521757, -77.781167), (12.16957, -68.990021), 
                                   (35.126413, 33.429859),   (49.817492, 15.472962),    (7.539989, -5.54708),    (56.26392, 9.501785), 
                                   (11.825138, 42.590275),   (15.414999, -61.370976),   (18.735693, -70.162651), (-1.831239, -78.183406), 
                                   (26.820553, 30.802498),   (13.794185, -88.89653),    (1.650801, 10.267895),   (15.179384, 39.782334), 
                                   (58.595272, 25.013607),   (-26.522503, 31.465866),   (9.145, 40.489673),      (-51.796253, -59.523613), 
                                   (61.892635, -6.911806),   (-16.578193, 179.414413),  (61.92411, 25.748151),   (46.227638, 2.213749), 
                                   (3.933889, -53.125782),   (-17.679742, -149.406843), (-49.280366, 69.348557), (-0.803689, 11.609444), 
                                   (13.443182, -15.310139),  (42.315407, 43.356892),    (51.165691, 10.451526),  (7.946527, -1.023194), 
                                   (36.137741, -5.345374),   (39.074208, 21.824312),    (71.706936, -42.604303), (12.262776, -61.604171), 
                                   (16.995971, -62.067641),  (13.444304, 144.793731),   (15.783471, -90.230759), (49.465691, -2.585278), 
                                   (9.945587, -9.696645),    (11.803749, -15.180413),   (4.860416, -58.93018),   (18.971187, -72.285215), 
                                   (-53.08181, 73.504158),   (41.902916, 12.453389),    (15.199999, -86.241905), (22.396428, 114.109497), 
                                   (47.162494, 19.503304),   (64.963051, -19.020835),   (20.593684, 78.96288),   (-0.789275, 113.921327), 
                                   (32.427908, 53.688046),   (33.223191, 43.679291),    (53.41291, -8.24389),    (54.236107, -4.548056), 
                                   (31.046051, 34.851612),   (41.87194, 12.56738),      (18.109581, -77.297508), (36.204824, 138.252924), 
                                   (49.214439, -2.13125),    (30.585164, 36.238414),    (48.019573, 66.923684),  (-0.023559, 37.906193), 
                                   (-3.370417, -168.734039), (40.339852, 127.510093),   (35.907757, 127.766922), (29.31166, 47.481766), 
                                   (41.20438, 74.766098),    (19.85627, 102.495496),    (56.879635, 24.603189),  (33.854721, 35.862285), 
                                   (-29.609988, 28.233608),  (6.428055, -9.429499),     (26.3351, 17.228331),    (47.166, 9.555373), 
                                   (55.169438, 23.881275),   (49.815273, 6.129583),     (22.198745, 113.543873), (-18.766947, 46.869107), 
                                   (-13.254308, 34.301525),  (4.210484, 101.975766),    (3.202778, 73.22068),    (17.570692, -3.996166), 
                                   (35.937496, 14.375416),   (7.131474, 171.184478),    (14.641528, -61.024174), (21.00789, -10.940835), 
                                   (-20.348404, 57.552152),  (-12.8275, 45.166244),     (23.634501, -102.552784),(7.425554, 150.550812), 
                                   (47.411631, 28.369885),   (43.750298, 7.412841),     (46.862496, 103.846656), (42.708678, 19.37439), 
                                   (16.742498, -62.187366),  (31.791702, -7.09262),     (-18.665695, 35.529562), (21.913965, 95.956223), 
                                   (-22.95764, 18.49041),    (-0.522778, 166.931503),   (28.394857, 84.124008),  (52.132633, 5.291266), 
                                   (-20.904305, 165.618042), (-40.900557, 174.885971),  (12.865416, -85.207229), (17.607789, 8.081666), 
                                   (9.081999, 8.675277),     (-19.054445, -169.867233), (-29.040835, 167.954712),(17.33083, 145.38469), 
                                   (60.472024, 8.468946),    (21.512583, 55.923255),    (30.375321, 69.345116),  (7.51498, 134.58252), 
                                   (31.952162, 35.233154),   (8.537981, -80.782127),    (-6.314993, 143.95555),  (-23.442503, -58.443832), 
                                   (-9.189967, -75.015152),  (12.879721, 121.774017),   (-24.703615, -127.439308), 
                                   (51.919438, 19.145136),   (39.399872, -8.224454),    (18.220833, -66.590149), (25.354826, 51.183884), 
                                   (41.608635, 21.745275),   (45.943161, 24.96676),     (61.52401, 105.318756),  (-1.940278, 29.873888), 
                                   (-21.115141, 55.536384),  (17.9, 62.8333),           (-24.143474, -10.030696),(17.357822, -62.782998), 
                                   (13.909444, -60.978893),  (18.073099, -63.082199),   (46.941936, -56.27111),  (12.984305, -61.287228), 
                                   (-13.759029, -172.104629),(43.94236, 12.457777),     (0.18636, 6.613081),     (23.885942, 45.079162), 
                                   (14.497401, -14.452362),  (44.016521, 21.005859),    (-4.679574, 55.491977),  (8.460555, -11.779889), 
                                   (1.352083, 103.819836),   (18.0425, 63.0548),        (48.669026, 19.699024),  (46.151241, 14.995463), 
                                   (-9.64571, 160.156194),   (5.152149, 46.199616),     (-30.559482, 22.937506), (-54.429579, -36.587909), 
                                   (6.877, 31.307),          (40.463667, -3.74922),     (7.873054, 80.771797),   (12.862807, 30.217636), 
                                   (3.919305, -56.027783),   (77.553604, 23.670272),    (60.128161, 18.643501),  (46.818188, 8.227512), 
                                   (34.802075, 38.996815),   (23.69781, 120.960515),    (38.861034, 71.276093),  (-6.369028, 34.888822), 
                                   (15.870032, 100.992541),  (-8.874217, 125.727539),   (8.619543, 0.824782),    (-8.967363, -171.855881), 
                                   (-21.178986, -175.198242),(10.691803, -61.222503),   (33.886917, 9.537499),   (38.963745, 35.243322), 
                                   (38.969719, 59.556278),   (21.694025, -71.797928),   (-7.109535, 177.64933),  (1.373333, 32.290275), 
                                   (48.379433, 31.16558),    (23.424076, 53.847818),    (55.378051, -3.435973),  (19.2823, 166.647), 
                                   (37.09024, -95.712891),   (-32.522779, -55.765835),  (41.377491, 64.585262),  (-15.376706, 166.959158), 
                                   (6.42375, -66.58973),     (14.058324, 108.277199),   (18.420695, -64.639968), (18.335765, -64.896335), 
                                   (-13.768752, -177.156097),(24.215527, -12.885834),   (15.552727, 48.516388), 
                                   (-13.133897, 27.849332),  (-19.015438, 29.154857),   (60.1785, 19.9156)
                                  ]
        self.color_names =        [ '#6929c4', '#9f1853', '#198038', '#b28600', '#8a3800', '#1192e8', '#fa4d56', '#002d9c', 
                                    '#009d9a', '#a56eff', '#005d5d', '#570408', '#ee538b', '#012749', '#da1e28', '#f1c21b', 
                                    '#ff832b', '#198038', '#bdd9bf', '#929084', '#ffc857', '#a997df', '#e5323b', '#2e4052', 
                                    '#e1daae', '#ff934f', '#cc2d35', '#058ed9', '#848fa2', '#2d3142', '#62a3f0', '#cc5f54', 
                                    '#e6cb60', '#523d02', '#c67ce6', '#00b524', '#4ad9bd', '#f53347', '#565c55',
                                    '#000000', '#ffff00', '#1ce6ff', '#ff34ff', '#ff4a46', '#008941', '#006fa6', '#a30059',
                                    '#ffdbe5', '#7a4900', '#0000a6', '#63ffac', '#b79762', '#004d43', '#8fb0ff', '#997d87',
                                    '#5a0007', '#809693', '#feffe6', '#1b4400', '#4fc601', '#3b5dff', '#4a3b53', '#ff2f80',
                                    '#61615a', '#ba0900', '#6b7900', '#00c2a0', '#ffaa92', '#ff90c9', '#b903aa', '#d16100',
                                    '#ddefff', '#000035', '#7b4f4b', '#a1c299', '#300018', '#0aa6d8', '#013349', '#00846f',
                                    '#372101', '#ffb500', '#c2ffed', '#a079bf', '#cc0744', '#c0b9b2', '#c2ff99', '#001e09',
                                    '#00489c', '#6f0062', '#0cbd66', '#eec3ff', '#456d75', '#b77b68', '#7a87a1', '#788d66',
                                    '#885578', '#fad09f', '#ff8a9a', '#d157a0', '#bec459', '#456648', '#0086ed', '#886f4c',
                                    '#34362d', '#b4a8bd', '#00a6aa', '#452c2c', '#636375', '#a3c8c9', '#ff913f', '#938a81',
                                    '#575329', '#00fecf', '#b05b6f', '#8cd0ff', '#3b9700', '#04f757', '#c8a1a1', '#1e6e00',
                                    '#7900d7', '#a77500', '#6367a9', '#a05837', '#6b002c', '#772600', '#d790ff', '#9b9700',
                                    '#549e79', '#fff69f', '#201625', '#72418f', '#bc23ff', '#99adc0', '#3a2465', '#922329',
                                    '#5b4534', '#fde8dc', '#404e55', '#0089a3', '#cb7e98', '#a4e804', '#324e72', '#6a3a4c',
                                    '#83ab58', '#001c1e', '#d1f7ce', '#004b28', '#c8d0f6', '#a3a489', '#806c66', '#222800',
                                    '#bf5650', '#e83000', '#66796d', '#da007c', '#ff1a59', '#8adbb4', '#1e0200', '#5b4e51',
                                    '#c895c5', '#320033', '#ff6832', '#66e1d3', '#cfcdac', '#d0ac94', '#7ed379', '#012c58',
                                    '#7a7bff', '#d68e01', '#353339', '#78afa1', '#feb2c6', '#75797c', '#837393', '#943a4d',
                                    '#b5f4ff', '#d2dcd5', '#9556bd', '#6a714a', '#001325', '#02525f', '#0aa3f7', '#e98176',
                                    '#dbd5dd', '#5ebcd1', '#3d4f44', '#7e6405', '#02684e', '#962b75', '#8d8546', '#9695c5',
                                    '#e773ce', '#d86a78', '#3e89be', '#ca834e', '#518a87', '#5b113c', '#55813b', '#e704c4',
                                    '#00005f', '#a97399', '#4b8160', '#59738a', '#ff5da7', '#f7c9bf', '#643127', '#513a01',
                                    '#6b94aa', '#51a058', '#a45b02', '#1d1702', '#e20027', '#e7ab63', '#4c6001', '#9c6966',
                                    '#64547b', '#97979e', '#006a66', '#391406', '#f4d749', '#0045d2', '#006c31', '#ddb6d0',
                                    '#7c6571', '#9fb2a4', '#00d891', '#15a08a', '#bc65e9', '#fffffe', '#c6dc99', '#203b3c',
                                    '#671190', '#6b3a64', '#f5e1ff', '#ffa0f2', '#ccaa35', '#374527', '#8bb400', '#797868',
                                    '#c6005a', '#3b000a', '#c86240', '#29607c', '#402334', '#7d5a44', '#ccb87c', '#b88183',
                                    '#aa5199', '#b5d6c3', '#a38469', '#9f94f0', '#a74571', '#b894a6', '#71bb8c', '#00b433',
                                    '#789ec9', '#6d80ba', '#953f00', '#5eff03', '#e4fffc', '#1be177', '#bcb1e5', '#76912f',
                                    '#003109', '#0060cd', '#d20096', '#895563', '#29201d', '#5b3213', '#a76f42', '#89412e',
                                    '#1a3a2a', '#494b5a', '#a88c85', '#f4abaa', '#a3f3ab', '#00c6c8', '#ea8b66', '#958a9f',
                                    '#bdc9d2', '#9fa064', '#be4700', '#658188', '#83a485', '#453c23', '#47675d', '#3a3f00',
                                    '#061203', '#dffb71', '#868e7e', '#98d058', '#6c8f7d', '#d7bfc2', '#3c3e6e', '#d83d66',
                                    '#2f5d9b', '#6c5e46', '#d25b88', '#5b656c', '#00b57f', '#545c46', '#866097', '#365d25',
                                    '#252f99', '#00ccff', '#674e60', '#fc009c', '#92896b', '#1e2324', '#dec9b2', '#9d4948',
                                    '#85abb4', '#342142', '#d09685', '#a4acac', '#00ffff', '#ae9c86', '#742a33', '#0e72c5',
                                    '#afd8ec', '#c064b9', '#91028c', '#feedbf', '#ffb789', '#9cb8e4', '#afffd1', '#2a364c',
                                    '#4f4a43', '#647095', '#34bbff', '#807781', '#920003', '#b3a5a7', '#018615', '#f1ffc8',
                                    '#976f5c', '#ff3bc1', '#ff5f6b', '#077d84', '#f56d93', '#5771da', '#4e1e2a', '#830055',
                                    '#02d346', '#be452d', '#00905e', '#be0028', '#6e96e3', '#007699', '#fec96d', '#9c6a7d',
                                    '#3fa1b8', '#893de3', '#79b4d6', '#7fd4d9', '#6751bb', '#b28d2d', '#e27a05', '#dd9cb8',
                                    '#aabc7a', '#980034', '#561a02', '#8f7f00', '#635000', '#cd7dae', '#8a5e2d', '#ffb3e1',
                                    '#6b6466', '#c6d300', '#0100e2', '#88ec69', '#8fccbe', '#21001c', '#511f4d', '#e3f6e3',
                                    '#ff8eb1', '#6b4f29', '#a37f46', '#6a5950', '#1f2a1a', '#04784d', '#101835', '#e6e0d0',
                                    '#ff74fe', '#00a45f', '#8f5df8', '#4b0059', '#412f23', '#d8939e', '#db9d72', '#604143',
                                    '#b5bace', '#989eb7', '#d2c4db', '#a587af', '#77d796', '#7f8c94', '#ff9b03', '#555196',
                                    '#31ddae', '#74b671', '#802647', '#2a373f', '#014a68', '#696628', '#4c7b6d', '#002c27',
                                    '#7a4522', '#3b5859', '#e5d381', '#fff3ff', '#679fa0', '#261300', '#2c5742', '#9131af',
                                    '#af5d88', '#c7706a', '#61ab1f', '#8cf2d4', '#c5d9b8', '#9ffffb', '#bf45cc', '#493941',
                                    '#863b60', '#b90076', '#003177', '#c582d2', '#c1b394', '#602b70', '#887868', '#babfb0',
                                    '#030012', '#d1acfe', '#7fdefe', '#4b5c71', '#a3a097', '#e66d53', '#637b5d', '#92bea5',
                                    '#00f8b3', '#beddff', '#3db5a7', '#dd3248', '#b6e4de', '#427745', '#598c5a', '#b94c59',
                                    '#8181d5', '#94888b', '#fed6bd', '#536d31', '#6eff92', '#e4e8ff', '#20e200', '#ffd0f2',
                                    '#4c83a1', '#bd7322', '#915c4e', '#8c4787', '#025117', '#a2aa45', '#2d1b21', '#a9ddb0',
                                    '#ff4f78', '#528500', '#009a2e', '#17fce4', '#71555a', '#525d82', '#00195a', '#967874',
                                    '#555558', '#0b212c', '#1e202b', '#efbfc4', '#6f9755', '#6f7586', '#501d1d', '#372d00',
                                    '#741d16', '#5eb393', '#b5b400', '#dd4a38', '#363dff', '#ad6552', '#6635af', '#836bba',
                                    '#98aa7f', '#464836', '#322c3e', '#7cb9ba', '#5b6965', '#707d3d', '#7a001d', '#6e4636',
                                    '#443a38', '#ae81ff', '#489079', '#897334', '#009087', '#da713c', '#361618', '#ff6f01',
                                    '#006679', '#370e77', '#4b3a83', '#c9e2e6', '#c44170', '#ff4526', '#73be54', '#c4df72',
                                    '#adff60', '#00447d', '#dccec9', '#bd9479', '#656e5b', '#ec5200', '#ff6ec2', '#7a617e',
                                    '#ddaea2', '#77837f', '#a53327', '#608eff', '#b599d7', '#a50149', '#4e0025', '#c9b1a9',
                                    '#03919a', '#1b2a25', '#e500f1', '#982e0b', '#b67180', '#e05859', '#006039', '#578f9b',
                                    '#305230', '#ce934c', '#b3c2be', '#c0bac0', '#b506d3', '#170c10', '#4c534f', '#224451',
                                    '#3e4141', '#78726d', '#b6602b', '#200441', '#ddb588', '#497200', '#c5aab6', '#033c61',
                                    '#71b2f5', '#a9e088', '#4979b0', '#a2c3df', '#784149', '#2d2b17', '#3e0e2f', '#57344c',
                                    '#0091be', '#e451d1', '#4b4b6a', '#5c011a', '#7c8060', '#ff9491', '#4c325d', '#005c8b',
                                    '#e5fda4', '#68d1b6', '#032641', '#140023', '#8683a9', '#cfff00', '#a72c3e', '#34475a',
                                    '#b1bb9a', '#b4a04f', '#8d918e', '#a168a6', '#813d3a', '#425218', '#da8386', '#776133',
                                    '#563930', '#8498ae', '#90c1d3', '#b5666b', '#9b585e', '#856465', '#ad7c90', '#e2bc00',
                                    '#e3aae0', '#b2c2fe', '#fd0039', '#009b75', '#fff46d', '#e87eac', '#dfe3e6', '#848590',
                                    '#aa9297', '#83a193', '#577977', '#3e7158', '#c64289', '#ea0072', '#c4a8cb', '#55c899',
                                    '#e78fcf', '#004547', '#f6e2e3', '#966716', '#378fdb', '#435e6a', '#da0004', '#1b000f',
                                    '#5b9c8f', '#6e2b52', '#011115', '#e3e8c4', '#ae3b85', '#ea1ca9', '#ff9e6b', '#457d8b',
                                    '#92678b', '#00cdbb', '#9ccc04', '#002e38', '#96c57f', '#cff6b4', '#492818', '#766e52',
                                    '#20370e', '#e3d19f', '#2e3c30', '#b2eace', '#f3bda4', '#a24e3d', '#976fd9', '#8c9fa8',
                                    '#7c2b73', '#4e5f37', '#5d5462', '#90956f', '#6aa776', '#dbcbf6', '#da71ff', '#987c95',
                                    '#52323c', '#bb3c42', '#584d39', '#4fc15f', '#a2b9c1', '#79db21', '#1d5958', '#bd744e',
                                    '#160b00', '#20221a', '#6b8295', '#00e0e4', '#102401', '#1b782a', '#daa9b5', '#b0415d',
                                    '#859253', '#97a094', '#06e3c4', '#47688c', '#7c6755', '#075c00', '#7560d5', '#7d9f00',
                                    '#c36d96', '#4d913e', '#5f4276', '#fce4c8', '#303052', '#4f381b', '#e5a532', '#706690',
                                    '#aa9a92', '#237363', '#73013e', '#ff9079', '#a79a74', '#029bdb', '#ff0169', '#c7d2e7',
                                    '#ca8869', '#80ffcd', '#bb1f69', '#90b0ab', '#7d74a9', '#fcc7db', '#99375b', '#00ab4d',
                                    '#abaed1', '#be9d91', '#e6e5a7', '#332c22', '#dd587b', '#f5fff7', '#5d3033', '#6d3800',
                                    '#ff0020', '#b57bb3', '#d7ffe6', '#c535a9', '#260009', '#6a8781', '#a8abb4', '#d45262',
                                    '#794b61', '#4621b2', '#8da4db', '#c7c890', '#6fe9ad', '#a243a7', '#b2b081', '#181b00',
                                    '#286154', '#4ca43b', '#6a9573', '#a8441d', '#5c727b', '#738671', '#d0cfcb', '#897b77',
                                    '#1f3f22', '#4145a7', '#da9894', '#a1757a', '#63243c', '#adaaff', '#00cde2', '#ddbc62',
                                    '#698eb1', '#208462', '#00b7e0', '#614a44', '#9bbb57', '#7a5c54', '#857a50', '#766b7e',
                                    '#014833', '#ff8347', '#7a8eba', '#274740', '#946444', '#ebd8e6', '#646241', '#373917',
                                    '#6ad450', '#81817b', '#d499e3', '#979440', '#011a12', '#526554', '#b5885c', '#a499a5',
                                    '#03ad89', '#b3008b', '#e3c4b5', '#96531f', '#867175', '#74569e', '#617d9f', '#e70452',
                                    '#067eaf', '#a697b6', '#b787a8', '#9cff93', '#311d19', '#3a9459', '#6e746e', '#b0c5ae',
                                    '#84edf7', '#ed3488', '#754c78', '#384644', '#c7847b', '#00b6c5', '#7fa670', '#c1af9e',
                                    '#2a7fff', '#72a58c', '#ffc07f', '#9debdd', '#d97c8e', '#7e7c93', '#62e674', '#b5639e',
                                    '#ffa861', '#c2a580', '#8d9c83', '#b70546', '#372b2e', '#0098ff', '#985975', '#20204c',
                                    '#ff6c60', '#445083', '#8502aa', '#72361f', '#9676a3', '#484449', '#ced6c2', '#3b164a',
                                    '#cca763', '#2c7f77', '#02227b', '#a37e6f', '#cde6dc', '#cdfffb', '#be811a', '#f77183',
                                    '#ede6e2', '#cdc6b4', '#ffe09e', '#3a7271', '#ff7b59', '#4e4e01', '#4ac684', '#8bc891',
                                    '#bc8a96', '#cf6353', '#dcde5c', '#5eaadd', '#f6a0ad', '#e269aa', '#a3dae4', '#436e83',
                                    '#002e17', '#ecfbff', '#a1c2b6', '#50003f', '#71695b', '#67c4bb', '#536eff', '#5d5a48',
                                    '#890039', '#969381', '#371521', '#5e4665', '#aa62c3', '#8d6f81', '#2c6135', '#410601',
                                    '#564620', '#e69034', '#6da6bd', '#e58e56', '#e3a68b', '#48b176', '#d27d67', '#b5b268',
                                    '#7f8427', '#ff84e6', '#435740', '#eae408', '#f4f5ff', '#325800', '#4b6ba5', '#adceff',
                                    '#9b8acc', '#885138', '#5875c1', '#7e7311', '#fea5ca', '#9f8b5b', '#a55b54', '#89006a',
                                    '#af756f', '#2a2000', '#576e4a', '#7f9eff', '#7499a1', '#ffb550', '#00011e', '#d1511c',
                                    '#688151', '#bc908a', '#78c8eb', '#8502ff', '#483d30', '#c42221', '#5ea7ff', '#785715',
                                    '#0cea91', '#fffaed', '#b3af9d', '#3e3d52', '#5a9bc2', '#9c2f90', '#8d5700', '#add79c',
                                    '#00768b', '#337d00', '#c59700', '#3156dc', '#944575', '#ecffdc', '#d24cb2', '#97703c',
                                    '#4c257f', '#9e0366', '#88ffec', '#b56481', '#396d2b', '#56735f', '#988376', '#9bb195',
                                    '#a9795c', '#e4c5d3', '#9f4f67', '#1e2b39', '#664327', '#afce78', '#322edf', '#86b487',
                                    '#c23000', '#abe86b', '#96656d', '#250e35', '#a60019', '#0080cf', '#caefff', '#323f61',
                                    '#a449dc', '#6a9d3b', '#ff5ae4', '#636a01', '#d16cda', '#736060', '#ffbaad', '#d369b4',
                                    '#ffded6', '#6c6d74', '#927d5e', '#845d70', '#5b62c1', '#2f4a36', '#e45f35', '#ff3b53',
                                    '#ac84dd', '#762988', '#70ec98', '#408543', '#2c3533', '#2e182d', '#323925', '#19181b',
                                    '#2f2e2c', '#023c32', '#9b9ee2', '#58afad', '#5c424d', '#7ac5a6', '#685d75', '#b9bcbd',
                                    '#834357', '#1a7b42', '#2e57aa', '#e55199', '#316e47', '#cd00c5', '#6a004d', '#7fbbec',
                                    '#f35691', '#d7c54a', '#62acb7', '#cba1bc', '#a28a9a', '#6c3f3b', '#ffe47d', '#dcbae3',
                                    '#5f816d', '#3a404a', '#7dbf32', '#e6ecdc', '#852c19', '#285366', '#b8cb9c', '#0e0d00',
                                    '#4b5d56', '#6b543f', '#e27172', '#0568ec', '#2eb500', '#d21656', '#efafff', '#682021',
                                    '#2d2011', '#da4cff', '#70968e', '#ff7b7d', '#4a1930', '#e8c282', '#e7dbbc', '#a68486',
                                    '#1f263c', '#36574e', '#52ce79', '#adaaa9', '#8a9f45', '#6542d2', '#00fb8c', '#5d697b',
                                    '#ccd27f', '#94a5a1', '#790229', '#e383e6', '#7ea4c1', '#4e4452', '#4b2c00', '#620b70',
                                    '#314c1e', '#874aa6', '#e30091', '#66460a', '#eb9a8b', '#eac3a3', '#98eab3', '#ab9180',
                                    '#b8552f', '#1a2b2f', '#94ddc5', '#9d8c76', '#9c8333', '#94a9c9', '#392935', '#8c675e',
                                    '#cce93a', '#917100', '#01400b', '#449896', '#1ca370', '#e08da7', '#8b4a4e', '#667776',
                                    '#4692ad', '#67bda8', '#69255c', '#d3bfff', '#4a5132', '#7e9285', '#77733c', '#e7a0cc',
                                    '#51a288', '#2c656a', '#4d5c5e', '#c9403a', '#ddd7f3', '#005844', '#b4a200', '#488f69',
                                    '#858182', '#d4e9b9', '#3d7397', '#cae8ce', '#d60034', '#aa6746', '#9e5585', '#ba6200',
                                    '#dee3E9', '#ebbaB5', '#fef3c7', '#a6e3d7', '#cbb4d5', '#808b96', '#f7dc6f', '#48c9b0',
                                    '#af7ac5', '#ec7063', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                                    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#bf77f6', '#ff9408', '#d1ffbd', '#c85a53',
                                    '#3a18b1', '#ff796c', '#04d8b2', '#ffb07c', '#aaa662', '#0485d1', '#fffe7a', '#b0dd16',
                                    '#d85679', '#12e193', '#82cafc', '#ac9362', '#f8481c', '#c292a1', '#c0fa8b', '#ca7b80',
                                    '#f4d054', '#fbdd7e', '#ffff7e', '#cd7584', '#f9bc08', '#c7c10c'
                                  ]
        self.data, self.entries = self.__read_bib(file_bib, db, del_duplicated)
        self.__make_bib()
    
    # Function: Prepare .bib File
    def __make_bib(self, verbose = True):
        self.ask_gpt_ap         = -1
        self.ask_gpt_bp         = -1
        self.ask_gpt_ep         = -1
        self.ask_gpt_ng         = -1
        self.ask_gpt_rt         = -1
        self.ask_gpt_sk         = -1
        self.ask_gpt_wd         = -1
        self.dy                 = pd.to_numeric(self.data['year'], downcast = 'float')
        self.date_str           = int(self.dy.min())
        self.date_end           = int(self.dy.max())
        self.doc_types          = self.data['document_type'].value_counts().sort_index()
        self.av_d_year          = self.dy.value_counts().sort_index()
        self.av_d_year          = round(self.av_d_year.mean(), 2)
        self.citation           = self.__get_citations(self.data['note'])
        self.av_c_doc           = round(sum(self.citation)/self.data.shape[0], 2)
        self.ref, self.u_ref    = self.__get_str(entry = 'references', s = ';',     lower = False, sorting = True)
        self.aut, self.u_aut    = self.__get_str(entry = 'author',     s = ' and ', lower = True,  sorting = True)
        self.aut_h              = self.__h_index()
        self.aut_docs           = [len(item) for item in self.aut]
        self.aut_single         = len([item  for item in self.aut_docs if item == 1])
        self.aut_multi          = [item for item in self.aut_docs if item > 1]
        self.aut_cit            = self.__get_counts(self.u_aut, self.aut, self.citation)
        self.kid, self.u_kid    = self.__get_str(entry = 'keywords', s = ';', lower = True, sorting = True)
        if ('unknow' in self.u_kid):
            self.u_kid.remove('unknow')
        self.kid_               = [item for sublist in self.kid for item in sublist]
        self.kid_count          = [self.kid_.count(item) for item in self.u_kid]
        idx                     = sorted(range(len(self.kid_count)), key = self.kid_count.__getitem__)
        idx.reverse()
        self.u_kid              = [self.u_kid[i] for i in idx]
        self.kid_count          = [self.kid_count[i] for i in idx]
        self.auk, self.u_auk    = self.__get_str(entry = 'author_keywords', s = ';', lower = True, sorting = True)
        if ('unknow' in self.u_auk):
            self.u_auk.remove('unknow')
        self.auk_               = [item for sublist in self.auk for item in sublist]
        self.auk_count          = [self.auk_.count(item) for item in self.u_auk]
        idx                     = sorted(range(len(self.auk_count)), key = self.auk_count.__getitem__)
        idx.reverse()
        self.u_auk              = [self.u_auk[i] for i in idx]
        self.auk_count          = [self.auk_count[i] for i in idx]
        self.jou, self.u_jou    = self.__get_str(entry = 'abbrev_source_title', s = ';', lower = True, sorting = True)
        if ('unknow' in self.u_jou):
            self.u_jou.remove('unknow')
        jou_                    = [item for sublist in self.jou for item in sublist]
        self.jou_count          = [jou_.count(item) for item in self.u_jou]
        idx                     = sorted(range(len(self.jou_count)), key = self.jou_count.__getitem__)
        idx.reverse()
        self.u_jou              = [self.u_jou[i] for i in idx]
        self.jou_count          = [self.jou_count[i] for i in idx]
        self.jou_cit            = self.__get_counts(self.u_jou, self.jou, self.citation)
        self.jou_cit            = self.__get_counts(self.u_jou, self.jou, self.citation)
        self.lan, self.u_lan    = self.__get_str(entry = 'language', s = '.', lower = True, sorting = True) 
        lan_                    = [item for sublist in self.lan for item in sublist]
        self.lan_count          = [lan_.count(item) for item in self.u_lan]
        self.ctr, self.u_ctr    = self.__get_countries()
        ctr_                    = [self.ctr[i][j] for i in range(0, len(self.aut)) for j in range(0, len(self.aut[i]))]
        self.ctr_count          = [ctr_.count(item) for item in self.u_ctr]
        self.ctr_cit            = self.__get_counts(self.u_ctr, self.ctr, self.citation)
        self.uni, self.u_uni    = self.__get_institutions() 
        uni_                    = [item for sublist in self.uni for item in sublist]
        self.uni_count          = [uni_.count(item) for item in self.u_uni]
        self.uni_cit            = self.__get_counts(self.u_uni,self.uni, self.citation)
        self.doc_aut            = self.__get_counts(self.u_aut, self.aut)
        self.av_doc_aut         = round(sum(self.doc_aut)/len(self.doc_aut), 2)
        self.t_c, self.s_c      = self.__total_and_self_citations()
        self.dy_ref             = self.__get_ref_year()
        self.natsort            = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]  
        #self.ordinal            = lambda n: '%d%s'%(n, {1: 'st', 2: 'nd', 3: 'rd'}.get(n if n < 20 else n % 10, 'th')) # [ordinal(n) for n in range(1, 15)]
        self.dy_c_year          = self.__get_collaboration_year()
        if ('UNKNOW' in self.u_ref):
            self.u_ref.remove('UNKNOW')
        self.__id_document()
        self.__id_author()
        self.__id_source()
        self.__id_institution()
        self.__id_country()
        self.__id_kwa()
        self.__id_kwp()
        #menambahkan ref journal
        self.__id_reff()
        if (verbose == True):
            for i in range(0, len(self.vb)):
                print(self.vb[i])
        return
    
    # Function: Document ID
    def __id_document(self):
        doc_list          = [str(i) for i in range(0, self.data.shape[0])]
        docs              = [self.data.loc[i, 'author']+' ('+self.data.loc[i, 'year']+'). '+self.data.loc[i, 'title']+'. '+self.data.loc[i, 'journal']+'. doi:'+self.data.loc[i, 'doi']+'. ' for i in range(0, self.data.shape[0])]
        self.table_id_doc = pd.DataFrame(zip(doc_list, docs), columns = ['ID', 'Document'])
        self.dict_id_doc  = dict(zip(doc_list, docs))
        return
    
    # Function: Author ID
    def __id_author(self):
        aut_list          = ['a_'+str(i) for i in range(0, len(self.u_aut))]
        self.table_id_aut = pd.DataFrame(zip(aut_list, self.u_aut), columns = ['ID', 'Author'])
        self.dict_id_aut  = dict(zip(aut_list, self.u_aut))
        self.dict_aut_id  = dict(zip(self.u_aut, aut_list))
        return
    
    # Function: Source ID
    def __id_source(self):
        jou_list          = ['j_'+str(i) for i in range(0, len(self.u_jou))]
        self.table_id_jou = pd.DataFrame(zip(jou_list, self.u_jou), columns = ['ID', 'Source'])
        self.dict_id_jou  = dict(zip(jou_list, self.u_jou))
        self.dict_jou_id  = dict(zip(self.u_jou, jou_list))
        return
    
    # Function: Institution ID
    def __id_institution(self):
        uni_list          = ['i_'+str(i) for i in range(0, len(self.u_uni))]
        self.table_id_uni = pd.DataFrame(zip(uni_list, self.u_uni), columns = ['ID', 'Institution'])
        self.dict_id_uni  = dict(zip(uni_list, self.u_uni))
        self.dict_uni_id  = dict(zip(self.u_uni, uni_list))
        return
    
    # Function: Country ID
    def __id_country(self):
        ctr_list          = ['c_'+str(i) for i in range(0, len(self.u_ctr))]
        self.table_id_ctr = pd.DataFrame(zip(ctr_list, self.u_ctr), columns = ['ID', 'Country'])
        self.dict_id_ctr  = dict(zip(ctr_list, self.u_ctr))
        self.dict_ctr_id  = dict(zip(self.u_ctr, ctr_list))
        return
    
    # Function: Authors' Keyword ID
    def __id_kwa(self):
        kwa_list          = ['k_'+str(i) for i in range(0, len(self.u_auk))]
        self.table_id_kwa = pd.DataFrame(zip(kwa_list, self.u_auk), columns = ['ID', 'KWA'])
        self.dict_id_kwa  = dict(zip(kwa_list, self.u_auk))
        self.dict_kwa_id  = dict(zip(self.u_auk, kwa_list))
        return
    
    # Function: References ID
    def __id_reff(self):
        reff_list          = ['r_'+str(i) for i in range(0, len(self.u_ref))]
        self.table_id_reff = pd.DataFrame(zip(reff_list, self.u_ref), columns = ['ID', 'Refences'])
        self.dict_id_reff  = dict(zip(reff_list, self.u_ref))
        self.dict_reff_id  = dict(zip(self.u_ref, reff_list))
        return
     
    def novelty_df(self):
        data = {
            'year': self.dy,
            'document': self.table_id_doc ['Document'],
            'journal': self.table_id_jou ['Source'],  # Assuming 'journal' corresponds to 'Source' in your code
            'keyword': self.table_id_kwa ['KWA']  # Assuming 'keyword' corresponds to 'KWA' in your code
        }

        df = pd.DataFrame(data)
        return df
        
    # Function: Keywords Plus ID
    def __id_kwp(self):
        kwp_list          = ['p_'+str(i) for i in range(0, len(self.u_kid))]
        self.table_id_kwp = pd.DataFrame(zip(kwp_list, self.u_kid), columns = ['ID', 'KWP'])
        self.dict_id_kwp  = dict(zip(kwp_list, self.u_kid))
        self.dict_kwp_id  = dict(zip(self.u_kid, kwp_list))
        return
    
    # Function: ID types
    def id_doc_types(self):
        dt     = self.doc_types.index.to_list()
        dt_ids = []
        for i in range(0, len(dt)):
            item = dt[i]
            idx  = self.data.index[self.data['document_type'] == item].tolist()
            dt_ids.append([item, idx])
        report_dt = pd.DataFrame(dt_ids, columns = ['Document Types', 'IDs'])
        return report_dt

    # Function: Filter
    def filter_bib(self, documents = [], doc_type = [], year_str = -1, year_end = -1, sources = [], core = -1, country = [], language = [], abstract = False):
        docs = []
        if (len(documents) > 0):
            self.data = self.data.iloc[documents, :]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (len(doc_type) > 0):
            for item in doc_type:
                if (sum(self.data['document_type'].isin([item])) > 0):
                    docs.append(item) 
                    self.data = self.data[self.data['document_type'].isin(docs)]
                    self.data = self.data.reset_index(drop = True)
                    self.__make_bib(verbose = False)
        if (year_str > -1):
            self.data = self.data[self.data['year'] >= str(year_str)]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (year_end > -1):
            self.data = self.data[self.data['year'] <= str(year_end)]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (len(sources) > 0):
            src_idx = []
            for source in sources:
                for i in range(0, len(self.jou)):
                    if (source == self.jou[i][0]):
                        src_idx.append(i)
            if (len(src_idx) > 0):
                self.data = self.data.iloc[src_idx, :]
                self.data = self.data.reset_index(drop = True)
                self.__make_bib(verbose = False)
        if (core == 1 or core == 2 or core == 3 or core == 12 or core == 23):
            key   = self.u_jou
            value = self.jou_count
            idx   = sorted(range(len(value)), key = value.__getitem__)
            idx.reverse()
            key   = [key[i]   for i in idx]
            value = [value[i] for i in idx]
            value = [sum(value[:i]) for i in range(1, len(value)+1)]
            c1    = int(value[-1]*(1/3))
            c2    = int(value[-1]*(2/3))
            if (core ==  1):
                key = [key[i] for i in range(0, len(key)) if value[i] <= c1]
            if (core ==  2):
                key = [key[i] for i in range(0, len(key)) if value[i] > c1 and value[i] <= c2]
            if (core ==  3):
               key = [key[i] for i in range(0, len(key)) if value[i] > c2]
            if (core == 12):
                key = [key[i] for i in range(0, len(key)) if value[i] <= c2]
            if (core == 23):
                key = [key[i] for i in range(0, len(key)) if value[i] > c1]
            sources   = self.data['abbrev_source_title'].str.lower()
            self.data = self.data[sources.isin(key)]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (len(country) > 0):
            ctr_idx   = [i for i in range(0, len(self.ctr)) if any(x in country for x in self.ctr[i])] 
            if (len(ctr_idx) > 0):
                self.data = self.data.iloc[ctr_idx, :]
                self.data = self.data.reset_index(drop = True)
                self.__make_bib(verbose = False)
        if (len(language) > 0):
            self.data = self.data[self.data['language'].isin(language)]
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        if (abstract == True):
            self.data = self.data[self.data['abstract'] != 'UNKNOW']
            self.data = self.data.reset_index(drop = True)
            self.__make_bib(verbose = False)
        self.__update_vb()
        self.__make_bib(verbose = True)
        return
    
