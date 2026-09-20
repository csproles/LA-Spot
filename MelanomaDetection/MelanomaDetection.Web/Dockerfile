# Place this file at: MelanomaDetection/MelanomaDetection.Web/Dockerfile

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
COPY . .
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
