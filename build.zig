const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ---------------------------------------------------------------------------
    // Common module — LLM helpers shared across ast-guidance source
    // ---------------------------------------------------------------------------
    const common_module = b.createModule(.{
        .root_source_file = b.path("src/common/llm.zig"),
        .target = target,
        .optimize = optimize,
    });

    // ---------------------------------------------------------------------------
    // ast-guidance executable
    // ---------------------------------------------------------------------------
    const guidance_exe = b.addExecutable(.{
        .name = "ast-guidance",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ast-guidance/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "common", .module = common_module },
            },
        }),
    });
    guidance_exe.linkLibC();
    b.installArtifact(guidance_exe);

    const run_guidance = b.addRunArtifact(guidance_exe);
    if (b.args) |args| {
        run_guidance.addArgs(args);
    }
    const run_step = b.step("run", "Run ast-guidance");
    run_step.dependOn(&run_guidance.step);

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------
    const guidance_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ast-guidance/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "common", .module = common_module },
            },
        }),
    });
    guidance_tests.linkLibC();

    const run_tests = b.addRunArtifact(guidance_tests);
    const test_step = b.step("test", "Run ast-guidance unit tests");
    test_step.dependOn(&run_tests.step);
}
