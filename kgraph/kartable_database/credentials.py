from sqlalchemy import create_engine

db_engine = create_engine("mysql://olivier_a:B5Dy9909Y4IiL27jqZ6Q" +
                          "@kartable-prod-read-replica.csuvfjuwwfxo.eu-west-1.rds.amazonaws.com/kartable")

available_courses = []
available_level = []
available_schoolyear = []

problematic_kd_ids = [53083, 53094, 53095, 3909, 3894, 4005, 50694]
