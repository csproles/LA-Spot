using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MelanomaDetection.Web.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddTelehealthScheduling : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<bool>(
                name: "IsProvider",
                table: "Users",
                type: "INTEGER",
                nullable: false,
                defaultValue: false);

            migrationBuilder.CreateTable(
                name: "Appointments",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    ProviderId = table.Column<Guid>(type: "TEXT", nullable: false),
                    PatientId = table.Column<Guid>(type: "TEXT", nullable: false),
                    StartUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    EndUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    Status = table.Column<int>(type: "INTEGER", nullable: false),
                    Reason = table.Column<string>(type: "TEXT", nullable: true),
                    MeetingId = table.Column<string>(type: "TEXT", maxLength: 100, nullable: true),
                    MeetingUrl = table.Column<string>(type: "TEXT", maxLength: 2048, nullable: true),
                    CreatedAtUtc = table.Column<DateTime>(type: "TEXT", nullable: false),
                    CancelledAtUtc = table.Column<DateTime>(type: "TEXT", nullable: true),
                    CancelledBy = table.Column<string>(type: "TEXT", maxLength: 20, nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Appointments", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "AvailabilityExceptions",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    ProviderId = table.Column<Guid>(type: "TEXT", nullable: false),
                    Date = table.Column<DateOnly>(type: "TEXT", nullable: false),
                    StartTime = table.Column<TimeOnly>(type: "TEXT", nullable: true),
                    EndTime = table.Column<TimeOnly>(type: "TEXT", nullable: true),
                    IsBlocked = table.Column<bool>(type: "INTEGER", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AvailabilityExceptions", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "AvailabilityRules",
                columns: table => new
                {
                    Id = table.Column<int>(type: "INTEGER", nullable: false)
                        .Annotation("Sqlite:Autoincrement", true),
                    ProviderId = table.Column<Guid>(type: "TEXT", nullable: false),
                    Weekday = table.Column<int>(type: "INTEGER", nullable: false),
                    StartTime = table.Column<TimeOnly>(type: "TEXT", nullable: false),
                    EndTime = table.Column<TimeOnly>(type: "TEXT", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AvailabilityRules", x => x.Id);
                });

            migrationBuilder.CreateTable(
                name: "Providers",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    Specialty = table.Column<string>(type: "TEXT", maxLength: 100, nullable: false),
                    AppointmentLengthMinutes = table.Column<int>(type: "INTEGER", nullable: false),
                    BufferMinutes = table.Column<int>(type: "INTEGER", nullable: false),
                    TimeZoneId = table.Column<string>(type: "TEXT", maxLength: 100, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_Providers", x => x.Id);
                });

            migrationBuilder.CreateIndex(
                name: "IX_Appointments_PatientId",
                table: "Appointments",
                column: "PatientId");

            migrationBuilder.CreateIndex(
                name: "IX_Appointments_ProviderId_StartUtc",
                table: "Appointments",
                columns: new[] { "ProviderId", "StartUtc" },
                unique: true,
                filter: "\"Status\" <> 1");

            migrationBuilder.CreateIndex(
                name: "IX_AvailabilityExceptions_ProviderId_Date",
                table: "AvailabilityExceptions",
                columns: new[] { "ProviderId", "Date" });

            migrationBuilder.CreateIndex(
                name: "IX_AvailabilityRules_ProviderId_Weekday",
                table: "AvailabilityRules",
                columns: new[] { "ProviderId", "Weekday" });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "Appointments");

            migrationBuilder.DropTable(
                name: "AvailabilityExceptions");

            migrationBuilder.DropTable(
                name: "AvailabilityRules");

            migrationBuilder.DropTable(
                name: "Providers");

            migrationBuilder.DropColumn(
                name: "IsProvider",
                table: "Users");
        }
    }
}
