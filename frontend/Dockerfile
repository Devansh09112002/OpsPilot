# OpsPilot frontend: build the static bundle, serve it with nginx.
# Only used for container-based hosting; on Render the frontend is deployed as
# a static site straight from `npm run build` output.

FROM node:22-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
# The API base URL is baked in at build time, so it must be supplied here.
ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL
RUN npm run build

FROM nginx:1.27-alpine AS runtime
COPY --from=build /app/dist /usr/share/nginx/html
COPY infra/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
