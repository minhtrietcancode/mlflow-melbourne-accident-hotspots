# Accident Dataset Documentation

## Overview
Table of road crash accidents that includes information about how the crash occurred, date, time and severity of the incident.

## Data Source
- **Download URL**: [accident.csv](https://opendata.transport.vic.gov.au/dataset/bb77800e-1857-4edc-bf9e-e188437a1c8e/resource/20772c1a-8b19-424a-a733-eb84f725f611/download/accident.csv)
- **Dataset Page**: [Victoria Road Crash Data - Accident](https://opendata.transport.vic.gov.au/dataset/victoria-road-crash-data/resource/20772c1a-8b19-424a-a733-eb84f725f611)

## Data Dictionary

| Field Name | Name | Type | Description | Fixed Values |
|------------|------|------|-------------|--------------|
| **ACCIDENT_NO** | Accident Number | STRING(12) | Primary Key for the database to uniquely identify the accident. Cannot contain NULL values. First character T indicates TIS incident and characters 2-5 typically represent the year in which the accident created in TIS system and characters 6-11 are a numeric sequencing number | - |
| **ACCIDENT DATE** | Accident Date | DATE(10) | Date that the accident occurred. The data is in yyyymmdd format in the database but appears in dd/mm/yyy format in the application. Field can contain null values. | - |
| **ACCIDENT TIME** | Accident Time | TIME(16) | Time in hh.mm.ss format. Original date stored in 24 hour format (i.e. 1pm = 1300 hours). Note the common practice used by the Police of 'rounding off the time' to the nearest 5 minutes or even nearest hour. | - |
| **ACCIDENT TYPE** | Accident Type | CHAR(1) | Character field indicates the type of accident. Basic description of what occurred, based on nine categories. | 1-9 |
| **ACCIDENT_TYPE_DESC** | Accident Type Description | VARCHAR(100) | Description of the accident type | 1=Collision with vehicle, 2=Struck pedestrian, 3=Struck animal, 4=Collision with a fixed object, 5=Collision with some other object, 6=Vehicle overturned (no collision), 7=Fall from or in moving vehicle, 8=No collision and no object struck, 9=Other accident |
| **DAY OF WEEK** | Day of Week | INTEGER(10) | The day of the week upon which the accident occurred | 1-7 |
| **DAY_WEEK_DESC** | Day of Week Description | NVARCHAR(30) | Description of the day of week | 1=Sunday, 2=Monday, 3=Tuesday, 4=Wednesday, 5=Thursday, 6=Friday, 7=Saturday |
| **DCA_CODE** | DCA Code | CHAR(3) | Definitions for Classifying Accidents. Cannot contain NULL values. | 100-781 |
| **DCA_DESC** | DCA Description | VARCHAR(100) | Detailed description for the Accident Classification | Various codes from 100-781 describing specific accident scenarios |
| **LIGHT CONDITION** | Light Condition | CHAR(1) | Light condition or level of brightness at the time of the accident. Cannot contain NULL values. | 1=Day, 2=Dusk/dawn, 3=Dark street lights on, 4=Dark street lights off, 5=Dark no street lights, 6=Dark street lights unknown, 9=Unknown |
| **NODE_ID** | Node ID | INTEGER(10) | The node id of the accident. Starts with 1 and incremented by one when a new accident location is identified. | - |
| **NO_OF_VEHICLES** | Number of Vehicles involved | INTEGER(10) | Number of vehicles involved in the accident. Includes bicycles but not objects, property, toys (skate boards), etc. | - |
| **NO_PERSONS_KILLED** | Number of Lives lost | INTEGER(10) | Number of people who have died in the crash | - |
| **NO_PERSONS_INJ_2** | Number of Person/s with Serious Injury | INTEGER(10) | Number of people with a serious injury | - |
| **NO_PERSONS_INJ_3** | Number of Person/s with other injury | INTEGER(10) | Number of people with an other injury | - |
| **NO_PERSONS_NOT_INJ** | Number of Person/s not injury | INTEGER(10) | Number of people with no injuries | - |
| **NO_PERSONS** | Number of Persons | INTEGER(10) | Number of people involved in the accident. NULL values are valid entries. | - |
| **POLICE_ATTEND** | Police attendance | CHAR(1) | Whether the police attended the scene of the accident or not. Cannot contain NULL values. | 1=Yes, 2=No, 9=Not known |
| **ROAD GEOMETRY** | Road Geometry | CHAR(1) | Code for layout of the road where the accident occurred | 1-9 |
| **ROAD_GEOMETRY_DESC** | Road Geometry Description | VARCHAR(100) | Descriptions of the layout of the road where the accident occurred | 1=Cross intersection, 2='T' Intersection, 3='Y' Intersection, 4=Multiple intersections, 5=Not at intersection, 6=Dead end, 7=Road closure, 8=Private property, 9=Unknown |
| **SEVERITY** | Severity | CHAR(1) | Estimation of the severity or seriousness of the accident | 1=Fatal accident, 2=Serious injury accident, 3=Other injury accident, 4=Non injury accident |
| **SPEED ZONE** | Speed Zone | CHAR(3) | Speed zone at the location of the accident. Generally assigned to the main vehicle involved. | 040=40 km/hr, 050=50 km/hr, 060=60 km/hr, 075=75 km/hr, 080=80 km/hr, 090=90 km/hr, 100=100 km/hr, 110=110 km/hr, 777=Other speed limit, 888=Camping grounds/off road, 999=Not known |

## Additional Information

| Field | Value |
|-------|-------|
| **Dataset Last Updated Date** | 11 August 2025 |
| **Last Updated Date** | 11 August 2025 |
| **Publication Date** | 1 January 2012 |
| **Format** | CSV |
| **License** | Creative Commons Attribution 4.0 |
| **Dataset File Size** | 30.3 MiB |
| **Dataset Security Value** | BIL1 OFFICIAL - Authorised Public Release |
| **Dataset Reporting Period Start** | 1 January 2012 |
| **Dataset Reporting Period End** | 31 December 2024 |
| **Geographic Coverage** | Victoria |

## Key Fields for Hotspot Analysis
- **NODE_ID**: Links to location coordinates in node.csv
- **ACCIDENT DATE**: For temporal analysis
- **LIGHT CONDITION**: Environmental factor
- **ROAD GEOMETRY**: Infrastructure factor
- **SPEED ZONE**: Road characteristics
- **SEVERITY**: Impact assessment