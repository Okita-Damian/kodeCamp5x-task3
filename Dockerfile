# Use official Node.js image
FROM node:24

# Set working directory inside container
WORKDIR /app

# Copy package files first (speeds up rebuilds)
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of your project files
COPY . .

# Expose port (same as in your code)
EXPOSE 8000

# Command to run the app
CMD ["npm", "run", "dev"]
