# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY App/frontend/package*.json ./
RUN npm ci --silent
COPY App/frontend/ .
RUN npm run build

# Production stage with nginx
FROM nginx:stable-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
