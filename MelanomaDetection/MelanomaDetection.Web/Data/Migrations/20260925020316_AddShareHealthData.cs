using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MelanomaDetection.Web.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddShareHealthData : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<bool>(
                name: "ShareHealthData",
                table: "Appointments",
                type: "INTEGER",
                nullable: false,
                defaultValue: false);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "ShareHealthData",
                table: "Appointments");
        }
    }
}
