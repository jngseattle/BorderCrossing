--
-- PostgreSQL database dump
--

-- Dumped from database version 9.4.4
-- Dumped by pg_dump version 9.4.0
-- Started on 2015-12-25 14:46:33 PST

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- TOC entry 191 (class 3079 OID 12123)
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- TOC entry 2413 (class 0 OID 0)
-- Dependencies: 191
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET search_path = public, pg_catalog;

--
-- TOC entry 172 (class 1259 OID 16519)
-- Name: crossing_id_seq; Type: SEQUENCE; Schema: public; Owner: jng
--

CREATE SEQUENCE crossing_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE crossing_id_seq OWNER TO jng;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- TOC entry 173 (class 1259 OID 16521)
-- Name: crossing; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE crossing (
    id integer DEFAULT nextval('crossing_id_seq'::regclass) NOT NULL,
    location_id integer NOT NULL,
    lane_id integer NOT NULL,
    direction_id integer NOT NULL
);


ALTER TABLE crossing OWNER TO jng;

--
-- TOC entry 174 (class 1259 OID 16525)
-- Name: crossingdata; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE crossingdata (
    date timestamp without time zone NOT NULL,
    crossing_id integer NOT NULL,
    waittime integer,
    volume integer,
    valid integer NOT NULL
);


ALTER TABLE crossingdata OWNER TO jng;

--
-- TOC entry 175 (class 1259 OID 16528)
-- Name: crossingdata_denorm; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE crossingdata_denorm (
    date timestamp without time zone NOT NULL,
    location_name character varying(45) NOT NULL,
    direction_name character varying(45) NOT NULL,
    lane_name character varying(45) NOT NULL,
    waittime integer,
    volume integer,
    valid integer NOT NULL
);


ALTER TABLE crossingdata_denorm OWNER TO jng;

--
-- TOC entry 183 (class 1259 OID 16621)
-- Name: datefeatures; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE datefeatures (
    date timestamp without time zone NOT NULL,
    year integer NOT NULL,
    month integer NOT NULL,
    dayofmonth integer NOT NULL,
    week integer NOT NULL,
    dayofweek integer NOT NULL,
    "time" time without time zone NOT NULL,
    hour integer NOT NULL,
    minofday integer NOT NULL,
    minute integer NOT NULL
);


ALTER TABLE datefeatures OWNER TO jng;

--
-- TOC entry 176 (class 1259 OID 16531)
-- Name: direction; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE direction (
    id integer NOT NULL,
    name character varying(45) NOT NULL
);


ALTER TABLE direction OWNER TO jng;

--
-- TOC entry 177 (class 1259 OID 16534)
-- Name: lane; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE lane (
    id integer NOT NULL,
    name character varying(45) NOT NULL
);


ALTER TABLE lane OWNER TO jng;

--
-- TOC entry 178 (class 1259 OID 16537)
-- Name: location; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE location (
    id integer NOT NULL,
    name character varying(45) NOT NULL,
    fullname character varying(45) NOT NULL
);


ALTER TABLE location OWNER TO jng;

--
-- TOC entry 179 (class 1259 OID 16540)
-- Name: location_direction; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE location_direction (
    location_id integer NOT NULL,
    direction_id integer NOT NULL
);


ALTER TABLE location_direction OWNER TO jng;

--
-- TOC entry 180 (class 1259 OID 16543)
-- Name: location_lane; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE location_lane (
    location_id integer NOT NULL,
    lane_id integer NOT NULL
);


ALTER TABLE location_lane OWNER TO jng;

--
-- TOC entry 188 (class 1259 OID 16781)
-- Name: model; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE model (
    name character varying(50) NOT NULL,
    notes character varying(1000),
    version character varying(10) NOT NULL
);


ALTER TABLE model OWNER TO jng;

--
-- TOC entry 186 (class 1259 OID 16709)
-- Name: mungedata; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE mungedata (
    munger_id integer NOT NULL,
    crossing_id integer NOT NULL,
    date timestamp without time zone NOT NULL,
    metric real,
    is_waittime boolean DEFAULT true NOT NULL
);


ALTER TABLE mungedata OWNER TO jng;

--
-- TOC entry 185 (class 1259 OID 16694)
-- Name: munger; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE munger (
    id integer NOT NULL,
    name character varying(100),
    description character varying(1000)
);


ALTER TABLE munger OWNER TO jng;

--
-- TOC entry 189 (class 1259 OID 16789)
-- Name: predictions; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE predictions (
    munger_id integer NOT NULL,
    date timestamp without time zone NOT NULL,
    waittime double precision NOT NULL,
    crossing_id integer NOT NULL,
    model_version character varying(20) NOT NULL
);


ALTER TABLE predictions OWNER TO jng;

--
-- TOC entry 184 (class 1259 OID 16674)
-- Name: publicholiday; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE publicholiday (
    date date NOT NULL,
    newyears boolean DEFAULT false NOT NULL,
    labor boolean DEFAULT false NOT NULL,
    us_mlk boolean DEFAULT false NOT NULL,
    us_washington boolean DEFAULT false NOT NULL,
    us_memorial boolean DEFAULT false NOT NULL,
    us_independence boolean DEFAULT false NOT NULL,
    us_columbus boolean DEFAULT false NOT NULL,
    us_veterans boolean DEFAULT false NOT NULL,
    us_thanksgiving boolean DEFAULT false NOT NULL,
    xmas boolean DEFAULT false NOT NULL,
    ca_goodfriday boolean DEFAULT false NOT NULL,
    ca_victoria boolean DEFAULT false NOT NULL,
    ca_canada boolean DEFAULT false NOT NULL,
    ca_civic boolean DEFAULT false NOT NULL,
    ca_thanksgiving boolean DEFAULT false NOT NULL,
    ca_family boolean DEFAULT false NOT NULL,
    mothers boolean DEFAULT false NOT NULL,
    halloween boolean DEFAULT false NOT NULL
);


ALTER TABLE publicholiday OWNER TO jng;

--
-- TOC entry 181 (class 1259 OID 16546)
-- Name: skiconditions; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE skiconditions (
    date timestamp without time zone NOT NULL,
    resort character varying(45) NOT NULL,
    snow24 integer,
    basedepth integer
);


ALTER TABLE skiconditions OWNER TO jng;

--
-- TOC entry 190 (class 1259 OID 16849)
-- Name: specialdates; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE specialdates (
    date date NOT NULL,
    locale character(2) NOT NULL,
    event character varying(30)
);


ALTER TABLE specialdates OWNER TO jng;

--
-- TOC entry 182 (class 1259 OID 16549)
-- Name: weather; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE weather (
    date timestamp without time zone NOT NULL,
    temp_max integer,
    temp_mean integer,
    temp_min integer,
    viz_max integer,
    viz_mean integer,
    viz_min integer,
    wind_max integer,
    wind_mean integer,
    wind_gust real,
    precip real,
    rain boolean DEFAULT false NOT NULL,
    snow boolean DEFAULT false NOT NULL,
    fog boolean DEFAULT false NOT NULL,
    thunderstorm boolean DEFAULT false NOT NULL,
    predicted boolean DEFAULT false NOT NULL
);


ALTER TABLE weather OWNER TO jng;

--
-- TOC entry 187 (class 1259 OID 16731)
-- Name: weather_raw; Type: TABLE; Schema: public; Owner: jng; Tablespace: 
--

CREATE TABLE weather_raw (
    pst character varying(30),
    max_temperaturef integer,
    mean_temperaturef integer,
    min_temperaturef integer,
    max_dew_pointf integer,
    meandew_pointf integer,
    min_dewpointf integer,
    max_humidity integer,
    mean_humidity integer,
    min_humidity integer,
    max_sea_level_pressurein double precision,
    mean_sea_level_pressurein double precision,
    min_sea_level_pressurein double precision,
    max_visibilitymiles integer,
    mean_visibilitymiles integer,
    min_visibilitymiles integer,
    max_wind_speedmph integer,
    mean_wind_speedmph integer,
    max_gust_speedmph integer,
    precipitationin double precision,
    cloudcover integer,
    events character varying(50),
    winddirdegrees integer
);


ALTER TABLE weather_raw OWNER TO jng;

--
-- TOC entry 2244 (class 2606 OID 16553)
-- Name: crossing_id_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY crossing
    ADD CONSTRAINT crossing_id_pkey PRIMARY KEY (id);


--
-- TOC entry 2249 (class 2606 OID 16558)
-- Name: crossingdata_date_crossing_id_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY crossingdata
    ADD CONSTRAINT crossingdata_date_crossing_id_pkey PRIMARY KEY (date, crossing_id);


--
-- TOC entry 2271 (class 2606 OID 16648)
-- Name: datefeatures_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY datefeatures
    ADD CONSTRAINT datefeatures_pkey PRIMARY KEY (date);


--
-- TOC entry 2251 (class 2606 OID 16561)
-- Name: direction_id_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY direction
    ADD CONSTRAINT direction_id_pkey PRIMARY KEY (id);


--
-- TOC entry 2254 (class 2606 OID 16564)
-- Name: lane_id_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY lane
    ADD CONSTRAINT lane_id_pkey PRIMARY KEY (id);


--
-- TOC entry 2262 (class 2606 OID 16571)
-- Name: location_direction_location_id_direction_id_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY location_direction
    ADD CONSTRAINT location_direction_location_id_direction_id_pkey PRIMARY KEY (location_id, direction_id);


--
-- TOC entry 2258 (class 2606 OID 16567)
-- Name: location_id_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY location
    ADD CONSTRAINT location_id_pkey PRIMARY KEY (id);


--
-- TOC entry 2265 (class 2606 OID 16574)
-- Name: location_lane_location_id_lane_id_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY location_lane
    ADD CONSTRAINT location_lane_location_id_lane_id_pkey PRIMARY KEY (location_id, lane_id);


--
-- TOC entry 2279 (class 2606 OID 16832)
-- Name: model_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY model
    ADD CONSTRAINT model_pkey PRIMARY KEY (version);


--
-- TOC entry 2277 (class 2606 OID 16848)
-- Name: mungedata_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY mungedata
    ADD CONSTRAINT mungedata_pkey PRIMARY KEY (munger_id, crossing_id, is_waittime, date);


--
-- TOC entry 2275 (class 2606 OID 16701)
-- Name: munger_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY munger
    ADD CONSTRAINT munger_pkey PRIMARY KEY (id);


--
-- TOC entry 2281 (class 2606 OID 16820)
-- Name: predictions_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY predictions
    ADD CONSTRAINT predictions_pkey PRIMARY KEY (model_version, munger_id, crossing_id, date);


--
-- TOC entry 2273 (class 2606 OID 16760)
-- Name: publicholiday_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY publicholiday
    ADD CONSTRAINT publicholiday_pkey PRIMARY KEY (date);


--
-- TOC entry 2267 (class 2606 OID 16577)
-- Name: skiconditions_date_resort_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY skiconditions
    ADD CONSTRAINT skiconditions_date_resort_pkey PRIMARY KEY (date, resort);


--
-- TOC entry 2283 (class 2606 OID 16855)
-- Name: specialdates_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY specialdates
    ADD CONSTRAINT specialdates_pkey PRIMARY KEY (date);


--
-- TOC entry 2269 (class 2606 OID 16579)
-- Name: weather_date_pkey; Type: CONSTRAINT; Schema: public; Owner: jng; Tablespace: 
--

ALTER TABLE ONLY weather
    ADD CONSTRAINT weather_date_pkey PRIMARY KEY (date);


--
-- TOC entry 2242 (class 1259 OID 16555)
-- Name: crossing_direction_id; Type: INDEX; Schema: public; Owner: jng; Tablespace: 
--

CREATE INDEX crossing_direction_id ON crossing USING btree (direction_id);


--
-- TOC entry 2245 (class 1259 OID 16554)
-- Name: crossing_lane_id; Type: INDEX; Schema: public; Owner: jng; Tablespace: 
--

CREATE INDEX crossing_lane_id ON crossing USING btree (lane_id);


--
-- TOC entry 2246 (class 1259 OID 16556)
-- Name: crossing_location_id; Type: INDEX; Schema: public; Owner: jng; Tablespace: 
--

CREATE INDEX crossing_location_id ON crossing USING btree (location_id);


--
-- TOC entry 2247 (class 1259 OID 16559)
-- Name: crossingdata_crossing_id; Type: INDEX; Schema: public; Owner: jng; Tablespace: 
--

CREATE INDEX crossingdata_crossing_id ON crossingdata USING btree (crossing_id);


--
-- TOC entry 2252 (class 1259 OID 16562)
-- Name: direction_name; Type: INDEX; Schema: public; Owner: jng; Tablespace: 
--

CREATE UNIQUE INDEX direction_name ON direction USING btree (name);


--
-- TOC entry 2255 (class 1259 OID 16565)
-- Name: lane_name; Type: INDEX; Schema: public; Owner: jng; Tablespace: 
--

CREATE UNIQUE INDEX lane_name ON lane USING btree (name);


--
-- TOC entry 2260 (class 1259 OID 16572)
-- Name: location_direction_direction_id; Type: INDEX; Schema: public; Owner: jng; Tablespace: 
--

CREATE INDEX location_direction_direction_id ON location_direction USING btree (direction_id);


--
-- TOC entry 2256 (class 1259 OID 16569)
-- Name: location_fullname; Type: INDEX; Schema: public; Owner: jng; Tablespace: 
--

CREATE UNIQUE INDEX location_fullname ON location USING btree (fullname);


--
-- TOC entry 2263 (class 1259 OID 16575)
-- Name: location_lane_lane_id; Type: INDEX; Schema: public; Owner: jng; Tablespace: 
--

CREATE INDEX location_lane_lane_id ON location_lane USING btree (lane_id);


--
-- TOC entry 2259 (class 1259 OID 16568)
-- Name: location_name; Type: INDEX; Schema: public; Owner: jng; Tablespace: 
--

CREATE UNIQUE INDEX location_name ON location USING btree (name);


--
-- TOC entry 2284 (class 2606 OID 16580)
-- Name: crossing_direction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: jng
--

ALTER TABLE ONLY crossing
    ADD CONSTRAINT crossing_direction_id_fkey FOREIGN KEY (direction_id) REFERENCES direction(id);


--
-- TOC entry 2285 (class 2606 OID 16585)
-- Name: crossing_lane_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: jng
--

ALTER TABLE ONLY crossing
    ADD CONSTRAINT crossing_lane_id_fkey FOREIGN KEY (lane_id) REFERENCES lane(id);


--
-- TOC entry 2286 (class 2606 OID 16590)
-- Name: crossing_location_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: jng
--

ALTER TABLE ONLY crossing
    ADD CONSTRAINT crossing_location_id_fkey FOREIGN KEY (location_id) REFERENCES location(id);


--
-- TOC entry 2287 (class 2606 OID 16595)
-- Name: crossingdata_crossing_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: jng
--

ALTER TABLE ONLY crossingdata
    ADD CONSTRAINT crossingdata_crossing_id_fkey FOREIGN KEY (crossing_id) REFERENCES crossing(id);


--
-- TOC entry 2288 (class 2606 OID 16600)
-- Name: location_direction_direction_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: jng
--

ALTER TABLE ONLY location_direction
    ADD CONSTRAINT location_direction_direction_id_fkey FOREIGN KEY (direction_id) REFERENCES direction(id);


--
-- TOC entry 2289 (class 2606 OID 16605)
-- Name: location_direction_location_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: jng
--

ALTER TABLE ONLY location_direction
    ADD CONSTRAINT location_direction_location_id_fkey FOREIGN KEY (location_id) REFERENCES location(id);


--
-- TOC entry 2290 (class 2606 OID 16610)
-- Name: location_lane_lane_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: jng
--

ALTER TABLE ONLY location_lane
    ADD CONSTRAINT location_lane_lane_id_fkey FOREIGN KEY (lane_id) REFERENCES lane(id);


--
-- TOC entry 2291 (class 2606 OID 16615)
-- Name: location_lane_location_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: jng
--

ALTER TABLE ONLY location_lane
    ADD CONSTRAINT location_lane_location_id_fkey FOREIGN KEY (location_id) REFERENCES location(id);


--
-- TOC entry 2293 (class 2606 OID 16719)
-- Name: mungedata_crossing_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: jng
--

ALTER TABLE ONLY mungedata
    ADD CONSTRAINT mungedata_crossing_id_fkey FOREIGN KEY (crossing_id) REFERENCES crossing(id);


--
-- TOC entry 2292 (class 2606 OID 16714)
-- Name: mungedata_munger_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: jng
--

ALTER TABLE ONLY mungedata
    ADD CONSTRAINT mungedata_munger_id_fkey FOREIGN KEY (munger_id) REFERENCES munger(id);


--
-- TOC entry 2294 (class 2606 OID 16821)
-- Name: predictions_crossing_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: jng
--

ALTER TABLE ONLY predictions
    ADD CONSTRAINT predictions_crossing_id_fkey FOREIGN KEY (crossing_id) REFERENCES crossing(id);


--
-- TOC entry 2296 (class 2606 OID 16833)
-- Name: predictions_model_version_fkey; Type: FK CONSTRAINT; Schema: public; Owner: jng
--

ALTER TABLE ONLY predictions
    ADD CONSTRAINT predictions_model_version_fkey FOREIGN KEY (model_version) REFERENCES model(version);


--
-- TOC entry 2295 (class 2606 OID 16826)
-- Name: predictions_munge_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: jng
--

ALTER TABLE ONLY predictions
    ADD CONSTRAINT predictions_munge_id_fkey FOREIGN KEY (munger_id) REFERENCES munger(id);


--
-- TOC entry 2412 (class 0 OID 0)
-- Dependencies: 5
-- Name: public; Type: ACL; Schema: -; Owner: jng
--

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON SCHEMA public FROM jng;
GRANT ALL ON SCHEMA public TO jng;
GRANT ALL ON SCHEMA public TO PUBLIC;


-- Completed on 2015-12-25 14:46:34 PST

--
-- PostgreSQL database dump complete
--

