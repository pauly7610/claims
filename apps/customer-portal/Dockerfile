FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Copy package files
COPY package.json pnpm-lock.yaml* ./
COPY apps/customer-portal/package.json ./apps/customer-portal/
COPY packages/design-system/package.json ./packages/design-system/
COPY packages/shared-types/package.json ./packages/shared-types/
COPY packages/api-client/package.json ./packages/api-client/

# Install dependencies
RUN npm install -g pnpm
RUN pnpm install --frozen-lockfile

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build packages first
RUN npm run build --workspace=@claims/design-system
RUN npm run build --workspace=@claims/shared-types
RUN npm run build --workspace=@claims/api-client

# Build the app
RUN npm run build --workspace=@claims/customer-portal

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/apps/customer-portal/public ./apps/customer-portal/public

# Set the correct permission for prerender cache
RUN mkdir apps/customer-portal/.next
RUN chown nextjs:nodejs apps/customer-portal/.next

# Automatically leverage output traces to reduce image size
COPY --from=builder --chown=nextjs:nodejs /app/apps/customer-portal/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/apps/customer-portal/.next/static ./apps/customer-portal/.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

CMD ["node", "apps/customer-portal/server.js"] 