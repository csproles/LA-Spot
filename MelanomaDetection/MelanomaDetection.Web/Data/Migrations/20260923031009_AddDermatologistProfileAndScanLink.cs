using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MelanomaDetection.Web.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddDermatologistProfileAndScanLink : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "Bio",
                table: "Providers",
                type: "TEXT",
                maxLength: 2000,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "Credentials",
                table: "Providers",
                type: "TEXT",
                maxLength: 200,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "LicenseNumber",
                table: "Providers",
                type: "TEXT",
                maxLength: 50,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "PhotoUrl",
                table: "Providers",
                type: "TEXT",
                maxLength: 2048,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "ScanExplanation",
                table: "Appointments",
                type: "TEXT",
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "ScanOverallVisualConcern",
                table: "Appointments",
                type: "TEXT",
                maxLength: 50,
                nullable: true);

            migrationBuilder.AddColumn<string>(
                name: "ScanProcessingId",
                table: "Appointments",
                type: "TEXT",
                maxLength: 100,
                nullable: true);

            migrationBuilder.AddColumn<double>(
                name: "ScanRiskScore",
                table: "Appointments",
                type: "REAL",
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "Bio",
                table: "Providers");

            migrationBuilder.DropColumn(
                name: "Credentials",
                table: "Providers");

            migrationBuilder.DropColumn(
                name: "LicenseNumber",
                table: "Providers");

            migrationBuilder.DropColumn(
                name: "PhotoUrl",
                table: "Providers");

            migrationBuilder.DropColumn(
                name: "ScanExplanation",
                table: "Appointments");

            migrationBuilder.DropColumn(
                name: "ScanOverallVisualConcern",
                table: "Appointments");

            migrationBuilder.DropColumn(
                name: "ScanProcessingId",
                table: "Appointments");

            migrationBuilder.DropColumn(
                name: "ScanRiskScore",
                table: "Appointments");
        }
    }
}
