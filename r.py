import pandas as pd


column_mapping = {
    # Tornado number, effectively an ID for this tornado in this year (integer)
    'om': 'tornado_id',

    # Year the tornado occurred, 1950-2022 (integer)
    'yr': 'year',

    # Month the tornado occurred, 1-12 (integer)
    'mo': 'month',

    # Day of the month the tornado occurred, 1-31 (integer)
    'dy': 'day',

    # Date of the tornado occurrence (date)
    'date': 'date',

    # Time of the tornado occurrence (time)
    'time': 'time',

    # Timezone of the occurrence, canonical tz database format (string)
    'tz': 'timezone',

    # Date and time normalized to UTC (datetime)
    'datetime_utc': 'datetime_utc',

    # State where the tornado occurred, two-letter postal abbreviation (string)
    'st': 'state_abbreviation',

    # State FIPS (Federal Information Processing Standards) code (integer)
    'stf': 'state_fips',

    # Magnitude of the tornado on the F scale (integer)
    'mag': 'magnitude',

    # Number of injuries caused by the tornado (integer)
    'inj': 'injuries',

    # Number of fatalities caused by the tornado (integer)
    'fat': 'fatalities',

    # Estimated property loss in dollars (double)
    'loss': 'property_loss',

    # Starting latitude of the tornado in decimal degrees (float)
    'slat': 'start_latitude',

    # Starting longitude of the tornado in decimal degrees (float)
    'slon': 'start_longitude',

    # Ending latitude of the tornado in decimal degrees (float)
    'elat': 'end_latitude',

    # Ending longitude of the tornado in decimal degrees (float)
    'elon': 'end_longitude',

    # Tornado track length in miles (float)
    'len': 'track_length_miles',

    # Tornado track width in yards (integer)
    'wid': 'track_width_yards',

    # Number of states affected by this tornado (integer)
    'ns': 'states_affected',

    # State number for the row (1 = entire track info, 0 = additional entry for same tornado) (integer)
    'sn': 'state_number',

    # FIPS code for the first county affected (integer)
    'f1': 'county_fips_1',

    # FIPS code for the second county affected (integer)
    'f2': 'county_fips_2',

    # FIPS code for the third county affected (integer)
    'f3': 'county_fips_3',

    # FIPS code for the fourth county affected (integer)
    'f4': 'county_fips_4',

    # Indicates if the magnitude was estimated (boolean)
    'fc': 'magnitude_estimated'
}


df = pd.read_csv('./tornados2.csv')

# Renaming the columns
df.rename(columns=column_mapping, inplace=True)

# save the file in new one with different name
df.to_csv('tornados_.csv', index=False)
