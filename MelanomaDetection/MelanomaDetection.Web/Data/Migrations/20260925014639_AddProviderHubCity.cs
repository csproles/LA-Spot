using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace MelanomaDetection.Web.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddProviderHubCity : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "HubCity",
                table: "Providers",
                type: "TEXT",
                maxLength: 50,
                nullable: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropColumn(
                name: "HubCity",
                table: "Providers");
        }
    }
}
