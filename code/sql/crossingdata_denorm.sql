CREATE VIEW `crossingdata_denorm` AS
    SELECT 
        location.name AS location_name,
        direction.name AS direction_name,
        lane.name AS lane_name,
        waittime,
        volume
    FROM
        crossingdata
            JOIN
        crossing ON crossing.id = crossing_id
            JOIN
        location ON location.id = crossing.location_id
            JOIN
        lane ON lane.id = crossing.lane_id
            JOIN
        direction ON direction.id = crossing.direction_id