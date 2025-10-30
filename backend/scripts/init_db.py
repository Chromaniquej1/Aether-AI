#!/usr/bin/env python3
"""
Database initialization script
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database.base import init_db, drop_db, engine
from backend.models import user, project, dataset, training_job, model, deployment
from sqlalchemy.orm import sessionmaker


def create_tables():
    """Create all database tables"""
    print("🔧 Creating database tables...")
    init_db()
    print("✅ Tables created successfully!")


def drop_tables():
    """Drop all database tables"""
    print("⚠️  Dropping all database tables...")
    response = input("Are you sure? This will delete all data! (yes/no): ")
    if response.lower() == 'yes':
        drop_db()
        print("✅ Tables dropped successfully!")
    else:
        print("❌ Operation cancelled")


def seed_data():
    """Seed database with sample data"""
    print("🌱 Seeding database with sample data...")
    
    Session = sessionmaker(bind=engine)
    db = Session()
    
    try:
        from backend.models.user import User, UserRole, SubscriptionTier
        
        demo_user = User(
            email="demo@modelforge.ai",
            username="demo",
            full_name="Demo User",
            hashed_password="$2b$12$dummy_hashed_password",
            is_active=True,
            is_verified=True,
            role=UserRole.USER,
            subscription_tier=SubscriptionTier.PRO,
            max_projects=10,
            max_gpu_hours_per_month=100
        )
        
        db.add(demo_user)
        db.commit()
        
        print(f"✅ Created demo user: {demo_user.email}")
        
        from backend.models.project import Project, TaskType
        
        sample_project = Project(
            name="Sample Image Classifier",
            description="A sample project for testing",
            task_type=TaskType.CLASSIFICATION,
            user_id=demo_user.id
        )
        
        db.add(sample_project)
        db.commit()
        
        print(f"✅ Created sample project: {sample_project.name}")
        print("\n📊 Database seeded successfully!")
        print(f"\n🔑 Demo credentials:")
        print(f"   Email: {demo_user.email}")
        print(f"   Use this email for testing")
        
    except Exception as e:
        print(f"❌ Error seeding database: {e}")
        db.rollback()
    
    finally:
        db.close()


def reset_database():
    """Drop and recreate all tables"""
    print("🔄 Resetting database...")
    response = input("This will delete all data. Continue? (yes/no): ")
    if response.lower() == 'yes':
        drop_db()
        init_db()
        print("✅ Database reset successfully!")
        
        response = input("Seed with sample data? (yes/no): ")
        if response.lower() == 'yes':
            seed_data()
    else:
        print("❌ Operation cancelled")


def show_tables():
    """Show all tables in database"""
    from sqlalchemy import inspect
    
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    print("\n📋 Database Tables:")
    print("="*50)
    for table in tables:
        print(f"  - {table}")
    print("="*50)
    print(f"Total: {len(tables)} tables\n")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║          🗄️  ModelForge-CV Database Setup 🗄️              ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Database management script")
    parser.add_argument(
        "command",
        choices=["create", "drop", "reset", "seed", "show"],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_tables()
    elif args.command == "drop":
        drop_tables()
    elif args.command == "reset":
        reset_database()
    elif args.command == "seed":
        seed_data()
    elif args.command == "show":
        show_tables()
    
    print("\n✨ Done!\n")