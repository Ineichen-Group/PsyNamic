import json
from django.core.management.base import BaseCommand, CommandParser
from psynamic.models import ClassGroup, LabelClass, Label

# python manage.py import_labels /home/vera/Documents/Arbeit/CRS/PsychNER/prodigy/choice_labels.json

class Command(BaseCommand):
    help = 'Import labels from a JSON file into the database'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('json_file', type=str)

    def handle(self, *args, **kwargs):
        json_file = kwargs['json_file']
        
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        for class_group_name, label_classes in data.items():
            # Create or get the ClassGroup
            class_group, created = ClassGroup.objects.get_or_create(name=class_group_name)

            for label_class_name, labels in label_classes.items():
                # Create or get the LabelClass
                label_class, created = LabelClass.objects.get_or_create(
                    name=label_class_name,
                    class_group=class_group
                )
                
                for label_name in labels:
                    # Create or get the Label
                    Label.objects.get_or_create(
                        name=label_name,
                        label_class=label_class
                    )
        
        self.stdout.write(self.style.SUCCESS('Successfully imported labels'))
