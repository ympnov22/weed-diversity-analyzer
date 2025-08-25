
CREATE DATABASE weed_diversity;
CREATE USER weed_user WITH ENCRYPTED PASSWORD 'weed_pass';
GRANT ALL PRIVILEGES ON DATABASE weed_diversity TO weed_user;

\c weed_diversity;

GRANT ALL ON SCHEMA public TO weed_user;
