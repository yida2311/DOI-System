# Generated by Django 3.0.7 on 2020-08-17 23:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('v0', '0006_auto_20200811_1922'),
    ]

    operations = [
        migrations.AddField(
            model_name='images',
            name='h',
            field=models.IntegerField(default=6000),
        ),
        migrations.AddField(
            model_name='images',
            name='step_col',
            field=models.FloatField(default=1000),
        ),
        migrations.AddField(
            model_name='images',
            name='step_ver',
            field=models.FloatField(default=1000),
        ),
        migrations.AddField(
            model_name='images',
            name='tile_cols',
            field=models.IntegerField(default=6),
        ),
        migrations.AddField(
            model_name='images',
            name='tile_vers',
            field=models.IntegerField(default=5),
        ),
        migrations.AddField(
            model_name='images',
            name='w',
            field=models.IntegerField(default=7000),
        ),
        migrations.AlterField(
            model_name='masks',
            name='keypoint_x1',
            field=models.IntegerField(default=1),
        ),
        migrations.AlterField(
            model_name='masks',
            name='keypoint_xf',
            field=models.IntegerField(default=3),
        ),
        migrations.AlterField(
            model_name='masks',
            name='keypoint_xt',
            field=models.IntegerField(default=2),
        ),
        migrations.AlterField(
            model_name='masks',
            name='keypoint_y1',
            field=models.IntegerField(default=1),
        ),
        migrations.AlterField(
            model_name='masks',
            name='keypoint_yf',
            field=models.IntegerField(default=3),
        ),
        migrations.AlterField(
            model_name='masks',
            name='keypoint_yt',
            field=models.IntegerField(default=2),
        ),
    ]
