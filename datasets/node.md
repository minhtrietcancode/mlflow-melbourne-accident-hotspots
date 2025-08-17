# Node Dataset Documentation

## Overview
Table contains information about where accident is located on the map. NODE_ID is the primary key of the table. NODE_ID is referred by ACCIDENT and FATAL_ACCIDENT table. Same node_id can be assigned to more than one accident.

## Data Source
- **Download URL**: [node.csv](https://opendata.transport.vic.gov.au/dataset/bb77800e-1857-4edc-bf9e-e188437a1c8e/resource/466fd3b5-201b-42b5-b10d-e926324fa215/download/node.csv)
- **Dataset Page**: [Victoria Road Crash Data - Node](https://opendata.transport.vic.gov.au/dataset/victoria-road-crash-data/resource/466fd3b5-201b-42b5-b10d-e926324fa215)

## Data Dictionary

| Field Name | Name | Type | Description | Fixed Values |
|------------|------|------|-------------|--------------|
| **ACCIDENT_NO** | Accident No | CHAR(12) | Primary Key for the database to uniquely identify the accident. Cannot contain NULL values. First character T indicates TIS incident and characters 2-5 typically represent the year in which the accident created in TIS system and characters 6-11 are a numeric sequencing number. | - |
| **NODE_ID** | Node ID | INTEGER(10) | The node id of the accident. Starts with 1 and incremented by one when a new accident location is identified. | - |
| **NODE_TYPE** | Node Type | NVARCHAR(1) | Character field indicates location type identified by the RCIS spatial system. Cannot contain NULL values. | I=Intersection, N=Non-Intersection, O=Off Road, U=Unknown |
| **AMG_X** | AMG X Coordinate | NUMERIC(38) | Decimal field that contains AMG coordinate X value. Cannot contain NULL values. Will have zero value for location unknown accidents. | - |
| **AMG_Y** | AMG Y Coordinate | NUMERIC(38) | Decimal field that contains AMG coordinate Y value. Cannot contain NULL values. Will have zero value for location unknown accidents. | - |
| **LGA_NAME** | LGA | NVARCHAR(100) | Character field contains the LGA (Local Government Area) name for the location of the crash. Cannot contain NULL values. | - |
| **DEG_URBAN_NAME** | Degree Urban Name | NVARCHAR(40) | Character field indicates degree of urban name for the location of the crash. Refers DEG_URBAN_NAME in DEGREE_OF_URBAN table. Cannot contain NULL values. | - |
| **LATITUDE** | Latitude | DECIMAL(10) | Latitude coordinate of the crash | - |
| **LONGITUDE** | Longitude | DECIMAL(10) | Longitude coordinate of the crash | - |
| **POSTCODE_CRASH** | Postcode Crash | INTEGER(10) | Postcode of the crash | - |

## Additional Information

| Field | Value |
|-------|-------|
| **Dataset Last Updated Date** | 11 August 2025 |
| **Last Updated Date** | 11 August 2025 |
| **Publication Date** | 1 January 2012 |
| **Format** | CSV |
| **License** | Creative Commons Attribution 4.0 |
| **Dataset File Size** | 20.5 MiB |
| **Dataset Security Value** | BIL1 OFFICIAL - Authorised Public Release |
| **Dataset Reporting Period Start** | 1 January 2012 |
| **Dataset Reporting Period End** | 31 December 2024 |
| **Geographic Coverage** | Victoria |

## Key Fields for Hotspot Analysis
- **NODE_ID**: Primary key to join with accident.csv
- **LATITUDE/LONGITUDE**: Essential for geographic mapping and clustering
- **LGA_NAME**: Filter for Melbourne locations
- **NODE_TYPE**: Intersection vs Non-intersection analysis
- **POSTCODE_CRASH**: Geographic segmentation
- **AMG_X/AMG_Y**: Alternative coordinate system (Australian Map Grid)

## Usage Notes
- Multiple accidents can share the same NODE_ID (same location, different incidents)
- Use LGA_NAME to filter for Melbourne-specific analysis
- LATITUDE/LONGITUDE are in decimal degrees format
- Zero coordinates indicate unknown locations