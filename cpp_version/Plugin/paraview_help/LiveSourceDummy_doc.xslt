<xml>
    <proxy>
        <group>sources</group>
        <name>EmulatedTimeDummySource</name>
        <label>EmulatedTimeDummySource</label>
        <documentation>
            <brief>Emulate dummy real time data using LiveSource behavior.</brief>
            <long/>
        </documentation>
        <property>
            <name>SourcePresets</name>
            <label>SourcePresets</label>
            <documentation>
                <brief/>
                <long/>
            </documentation>
            <defaults>0</defaults>
            <domains>
                <domain>
                    <text>The value(s) is an enumeration of the following:</text>
                    <list>
                        <item>Sphere Preset (0)</item>
                        <item>Cone Preset (1)</item>
                        <item>Cube Preset (2)</item>
                    </list>
                </domain>
            </domains>
        </property>
        <property>
            <name>TimestepValues</name>
            <label>TimestepValues</label>
            <defaults/>
            <domains/>
        </property>
    </proxy>
    <proxy>
        <group>sources</group>
        <name>LiveSourceDummy</name>
        <label>Live Souce (dummy)</label>
        <documentation>
            <brief>Live source dummy.</brief>
            <long/>
        </documentation>
        <property>
            <name>MaxIterations</name>
            <label>MaxIterations</label>
            <documentation>
                <brief/>
                <long>
          The number of iterations before the live source stop.
        </long>
            </documentation>
            <defaults>2000</defaults>
            <domains/>
        </property>
    </proxy>
</xml>
