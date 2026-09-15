# Place this file at: MelanomaDetection/MelanomaDetection.Web/Dockerfile

FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src
COPY . .
RUN dotnet publish -c Release -o /app

FROM mcr.microsoft.com/dotnet/aspnet:10.0
WORKDIR /app
COPY --from=build /app .

# Plain HTTP inside the container — avoids needing dev certs trusted in-container.
# We still expose it to the host as localhost:7001 via the port mapping in docker-compose.yml.
ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

ENTRYPOINT ["dotnet", "MelanomaDetection.Web.dll"]
