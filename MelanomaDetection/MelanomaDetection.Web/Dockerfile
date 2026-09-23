# Place this file at: MelanomaDetection/MelanomaDetection.Web/Dockerfile

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build

# Node, to run the Tailwind CLI (see MelanomaDetection.Web.csproj's TailwindBuild target).
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN npm ci
# Run Tailwind as its own step, not just via the .csproj's TailwindBuild target: MSBuild's
# static-web-assets manifest is built from a glob evaluated at project-load time, before any
# target runs, so a file a target generates mid-build is invisible to that same build's
# manifest (confirmed: wwwroot/css/tailwind.css existed on disk but had zero routes in the
# endpoints.json until a second, separate build saw it already present at evaluation time).
# Generating it here, before publish's evaluation even starts, is what actually avoids that.
RUN npx @tailwindcss/cli -i Styles/tailwind.css -o wwwroot/css/tailwind.css --minify
RUN dotnet publish -c Release -o /app

FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /app .
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

# Runs as the image's unprivileged "app" user. The entrypoint starts as root only
# long enough to hand /app/data (accounts database + cookie keys) to that user --
# volumes made by an earlier root-run version are otherwise unwritable -- then
# drops privileges.
RUN mkdir -p /app/data \
    && chown -R app:app /app/data \
    && chmod +x /app/docker-entrypoint.sh

# Plain HTTP inside the container — avoids needing dev certs trusted in-container.
# We still expose it to the host as localhost:7001 via the port mapping in docker-compose.yml.
ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["dotnet", "MelanomaDetection.Web.dll"]
