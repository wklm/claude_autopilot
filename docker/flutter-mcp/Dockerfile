FROM node:20-alpine

# Install git for npm dependencies
RUN apk add --no-cache git

# Create app directory
WORKDIR /app

# Install Flutter MCP globally
RUN npm install -g flutter-mcp

# Use the existing node user (UID 1000) instead of creating a new one
USER node

# Expose HTTP port
EXPOSE 8000

# Set default environment variables
ENV MCP_TRANSPORT=http
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000

# Run Flutter MCP server in HTTP mode
CMD ["flutter-mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]